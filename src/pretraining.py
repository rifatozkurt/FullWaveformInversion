import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import io
from src import metrics
from src import networks as NN
from src.experiments.base import get_device, simulation_parameters


def _save_pretraining_outputs(
    run_dir,
    training_loss,
    validation_loss,
    validation_gamma_mse,
    validation_dice,
    validation_iou,
):
    run_dir = Path(run_dir)
    histories_dir = io.ensure_dir(run_dir / "histories")
    figures_dir = io.ensure_dir(run_dir / "figures")
    outputs_dir = io.ensure_dir(run_dir / "outputs")

    training_path = histories_dir / "pretraining_training_loss_history.txt"
    validation_path = histories_dir / "pretraining_validation_loss_history.txt"
    gamma_mse_path = histories_dir / "pretraining_validation_gamma_mse_history.txt"
    dice_path = histories_dir / "pretraining_validation_dice_history.txt"
    iou_path = histories_dir / "pretraining_validation_iou_history.txt"
    np.savetxt(training_path, training_loss, delimiter=", ")
    np.savetxt(validation_path, validation_loss, delimiter=", ")
    np.savetxt(gamma_mse_path, validation_gamma_mse, delimiter=", ")
    np.savetxt(dice_path, validation_dice, delimiter=", ")
    np.savetxt(iou_path, validation_iou, delimiter=", ")

    metrics_csv_path = histories_dir / "pretraining_validation_metrics.csv"
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "gamma_mse", "dice_score", "iou"],
        )
        writer.writeheader()
        for epoch, (gamma_mse, dice, iou_value) in enumerate(
            zip(validation_gamma_mse, validation_dice, validation_iou),
            start=1,
        ):
            writer.writerow(
                {
                    "epoch": epoch,
                    "gamma_mse": float(gamma_mse),
                    "dice_score": float(dice),
                    "iou": float(iou_value),
                }
            )

    plot_path = figures_dir / "pretraining_loss_history.png"
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    epochs = np.arange(1, len(training_loss) + 1)
    ax.plot(epochs, training_loss, label="Training loss", color="#2f5aa8", linewidth=2)
    ax.plot(epochs, validation_loss, label="Validation loss", color="#b64040", linewidth=2)
    ax.set_title("U-Net Pretraining Loss History")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Native Loss (0.5 × Padded Gamma MSE)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    np.savez(
        outputs_dir / "pretraining_loss_history.npz",
        training_loss=training_loss,
        validation_loss=validation_loss,
        validation_gamma_mse=validation_gamma_mse,
        validation_dice=validation_dice,
        validation_iou=validation_iou,
    )
    return {
        "training_loss_path": training_path,
        "validation_loss_path": validation_path,
        "validation_metrics_path": metrics_csv_path,
        "plot_path": plot_path,
    }


def pretrain_unet(config, data_dir=None, output_dir=None, progress_callback=None, run_dir=None):
    """
    Train the U-Net from initial adjoint gradients to true gamma fields.
    Save model weights and return the saved model path.
    """
    params = simulation_parameters(config)
    cfg = config["pretraining"]
    unet_cfg = config.get("models", {}).get("unet", {})
    device = get_device()

    Nx = params["Nx"]
    Ny = params["Ny"]
    gamma0 = params["gamma0"]

    destinationFolder = Path(data_dir or config["paths"]["train_data"])
    output_dir = Path(output_dir or config["paths"]["pretrained_models"])
    io.ensure_dir(output_dir)

    numberOfSamples = int(cfg["numberOfSamples"])
    # Legacy configs preload the complete dataset to the training device. Large
    # comparison runs can opt into CPU storage and move only each mini-batch.
    preload_to_device = bool(cfg.get("preload_to_device", True))
    storage_device = device if preload_to_device else torch.device("cpu")
    initialGradient = torch.zeros(
        (numberOfSamples, 1, Nx + 1, Ny + 1),
        dtype=torch.float32,
        device=storage_device,
    )
    gamma = torch.ones(
        (numberOfSamples, 1, Nx + 3, Ny + 3),
        dtype=torch.float32,
        device=storage_device,
    )
    if "sample_ids" in cfg:
        idx_numberOfSamples = [int(item) for item in cfg["sample_ids"][:numberOfSamples]]
        if len(idx_numberOfSamples) != numberOfSamples:
            raise ValueError("pretraining.sample_ids must contain at least numberOfSamples entries")
    else:
        idx_numberOfSamples = random.sample(range(int(cfg["availableSamples"])), numberOfSamples)

    print(f"Loading {numberOfSamples} pretraining sample(s) from {destinationFolder}", flush=True)
    load_print_every = max(1, numberOfSamples // 20)
    for idx, file_idx in enumerate(idx_numberOfSamples):
        if idx == 0 or (idx + 1) % load_print_every == 0 or idx + 1 == numberOfSamples:
            print(
                f"Loading sample {idx + 1}/{numberOfSamples}: material{file_idx}.h5, gradient{file_idx}.h5",
                flush=True,
            )
        initialGradient[idx, 0] = torch.tensor(
            io.load_hdf(destinationFolder / f"gradient{file_idx}.h5")
        ).to(storage_device).to(torch.float32)

        gamma[idx, 0, 1:-1, 1:-1] = torch.tensor(
            io.load_hdf(destinationFolder / f"material{file_idx}.h5")
        ).to(storage_device).to(torch.float32)

    trainingType = cfg["trainingType"]
    NNchannels = unet_cfg.get("channels", cfg["NNchannels"])
    numberOfConvolutionsPerBlock = int(
        unet_cfg.get(
            "number_of_convolutions_per_block",
            cfg["numberOfConvolutionsPerBlock"],
        )
    )
    batch_norm = bool(unet_cfg.get("batch_norm", True))
    # Legacy rule (legacy/Pretraining.py:74): batchSize = numberOfSamples // 10,
    # and validation runs as ONE full-set batch. Both are reproduced exactly at
    # the 800-sample scale. `max_batch_size` only caps the large scaling runs,
    # where the legacy rule would ask for a 1500-sample batch.
    legacy_batch_size = max(1, numberOfSamples // int(cfg["batchDivisor"]))
    batchSize = max(1, int(cfg.get("batch_size", legacy_batch_size)))
    batch_cap = cfg.get("max_batch_size")
    if batch_cap is not None:
        batchSize = max(1, min(batchSize, int(batch_cap)))

    # Shared normalization -- identical to the SegFormer's, so the comparison
    # isolates architecture rather than preprocessing. See src/networks.py.
    normalization = dict(config.get("gradient_normalization", {}) or {})
    inputData = NN.normalize_gradient(initialGradient, **normalization)

    dataset = NN.FWIDataset(inputData, gamma, device)
    datasetTraining, datasetValidation = torch.utils.data.random_split(
        dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(int(cfg["split_seed"]))
    )

    batchSize = min(batchSize, max(1, len(datasetTraining)))
    validation_batch_size = max(
        1,
        int(cfg.get("validation_batch_size", len(datasetValidation))),
    )
    validation_cap = cfg.get("max_validation_batch_size")
    if validation_cap is not None:
        validation_batch_size = max(1, min(validation_batch_size, int(validation_cap)))
    validation_batch_size = min(validation_batch_size, max(1, len(datasetValidation)))
    dataloaderTraining = DataLoader(datasetTraining, batch_size=batchSize)
    dataloaderValidation = DataLoader(
        datasetValidation,
        batch_size=validation_batch_size,
    )
    number_training_batches = len(dataloaderTraining)
    if run_dir is not None:
        outputs_dir = io.ensure_dir(Path(run_dir) / "outputs")
        np.savetxt(outputs_dir / "pretraining_sample_ids.txt", idx_numberOfSamples, fmt="%d")
        np.savetxt(outputs_dir / "pretraining_training_subset_indices.txt", datasetTraining.indices, fmt="%d")
        np.savetxt(outputs_dir / "pretraining_validation_subset_indices.txt", datasetValidation.indices, fmt="%d")
        sample_ids = np.asarray(idx_numberOfSamples, dtype=np.int64)
        np.savetxt(
            outputs_dir / "pretraining_training_sample_ids.txt",
            sample_ids[np.asarray(datasetTraining.indices, dtype=np.int64)],
            fmt="%d",
        )
        np.savetxt(
            outputs_dir / "pretraining_validation_sample_ids.txt",
            sample_ids[np.asarray(datasetValidation.indices, dtype=np.int64)],
            fmt="%d",
        )
    print(
        "Pretraining setup: epochs={}, batch_size={}, training_batches={}, "
        "validation_batch_size={}, validation_samples={}, storage_device={}".format(
            int(cfg["epochs"]),
            batchSize,
            number_training_batches,
            validation_batch_size,
            len(datasetValidation),
            storage_device,
        ),
        flush=True,
    )

    torch.manual_seed(int(cfg["seed"]))
    model_type = cfg["model_type"]
    torch.use_deterministic_algorithms(True)
    model = NN.Unet(
        NNchannels,
        numberOfConvolutionsPerBlock,
        gamma0,
        bnorm=batch_norm,
    )
    NN.initWeights(model)
    torch.nn.init.normal_(model.convolutionsUp[-1].weight, std=0.01, mean=0.7)
    model.convolutionsUp[-1].bias.data.fill_(3)
    model.to(device)

    lr = cfg["lr"]
    alpha = cfg["alpha"]
    beta = cfg["beta"]
    epochs = int(cfg["epochs"])
    clipGrad = cfg["clipGrad"]

    optimizer = torch.optim.RMSprop(model.parameters(), lr)
    lr_lambda = lambda epoch: (beta * epoch + 1) ** alpha
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trainingCostHistory = np.zeros(epochs)
    trainingMSEHistory = np.zeros(epochs)
    validationCostHistory = np.zeros(epochs)
    validationGammaMSEHistory = np.zeros(epochs)
    validationDiceHistory = np.zeros(epochs)
    validationIoUHistory = np.zeros(epochs)
    print_every_batches = max(1, int(cfg.get("print_every_batches", 1)))
    start = time.perf_counter()
    # Legacy compatibility: Pretraining.py resets beta to zero after creating
    # the scheduler lambda. Because the lambda closes over beta, the original
    # learning-rate schedule is effectively constant even though cfg["beta"] is
    # 0.2. Keep this quirk so UI/CLI pretraining matches the legacy script.
    beta = 0

    for epoch in range(epochs):
        model.train()
        for batch, sample in enumerate(dataloaderTraining):
            sample[0] = sample[0].to(device)
            sample[1] = sample[1].to(device)
            optimizer.zero_grad(set_to_none=True)

            gammaPred = torch.ones((len(sample[0]), 1, Nx + 3, Ny + 3), device=device, dtype=torch.float32)
            gammaPred[:, :, 1:-1, 1:-1] = model(sample[0])
            cost = 0.5 * torch.mean((gammaPred - sample[1]) ** 2)
            cost.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clipGrad)
            optimizer.step()
            scheduler.step()
            trainingCostHistory[epoch] += cost.detach().cpu()
            if (
                batch == 0
                or batch + 1 == number_training_batches
                or (batch + 1) % print_every_batches == 0
            ):
                print(
                    "Epoch {}/{} batch {}/{} training_cost={:.6E}".format(
                        epoch + 1,
                        epochs,
                        batch + 1,
                        number_training_batches,
                        float(cost.detach().cpu()),
                    ),
                    flush=True,
                )

        trainingCostHistory[epoch] /= batch + 1
        trainingMSEHistory[epoch] /= batch + 1

        model.eval()
        validation_cost_sum = 0.0
        validation_gamma_mse_sum = 0.0
        validation_samples = 0
        validation_tp = 0.0
        validation_fp = 0.0
        validation_fn = 0.0
        with torch.no_grad():
            for sample in dataloaderValidation:
                sample[0] = sample[0].to(device)
                sample[1] = sample[1].to(device)
                gamma_interior_pred = model(sample[0])
                gamma_interior_target = sample[1][:, :, 1:-1, 1:-1]
                gammaPred = torch.ones(
                    (len(sample[0]), 1, Nx + 3, Ny + 3),
                    device=device,
                    dtype=torch.float32,
                )
                gammaPred[:, :, 1:-1, 1:-1] = gamma_interior_pred
                # The training loss keeps its 0.5 (it mirrors the FWI misfit
                # convention); the REPORTED gamma error does not -- see
                # src/metrics.py for why the two were split.
                batch_validation_cost = 0.5 * torch.mean((gammaPred - sample[1]) ** 2)
                validation_cost_sum += (
                    float(batch_validation_cost.detach().cpu()) * len(sample[0])
                )
                validation_gamma_mse_sum += (
                    metrics.gamma_mse(gamma_interior_pred, gamma_interior_target)
                    * len(sample[0])
                )
                validation_samples += len(sample[0])

                # Void masks are thresholded on gamma directly, identically for
                # every model, via the shared helper in src/metrics.py.
                predicted_void = metrics.void_mask(gamma_interior_pred)
                target_void = metrics.void_mask(gamma_interior_target)
                validation_tp += float(
                    (predicted_void & target_void).sum().detach().cpu()
                )
                validation_fp += float(
                    (predicted_void & ~target_void).sum().detach().cpu()
                )
                validation_fn += float(
                    (~predicted_void & target_void).sum().detach().cpu()
                )
        validationCostHistory[epoch] = validation_cost_sum / max(1, validation_samples)
        validationGammaMSEHistory[epoch] = (
            validation_gamma_mse_sum / max(1, validation_samples)
        )
        metric_eps = 1e-8
        validationDiceHistory[epoch] = (
            2.0 * validation_tp
            / (2.0 * validation_tp + validation_fp + validation_fn + metric_eps)
        )
        validationIoUHistory[epoch] = (
            validation_tp
            / (validation_tp + validation_fp + validation_fn + metric_eps)
        )

        if epoch % 10 == 0:
            elapsed_time = time.perf_counter() - start
            string = "Epoch: {}/{}\tTraining Cost: {:.6E}\t Validation Cost: {:3E}\tElapsed time: {:2f} \t Input per sec: {:3f}"
            print(string.format(epoch, epochs - 1, trainingCostHistory[epoch], validationCostHistory[epoch], elapsed_time, batchSize / elapsed_time))
            start = time.perf_counter()

        if progress_callback is not None:
            progress_callback(
                epoch,
                epochs,
                float(trainingCostHistory[epoch]),
                float(validationCostHistory[epoch]),
            )

    path = output_dir / (
        "model_" + model_type + "_" + str(epochs) + "_" + trainingType + "_"
        + str(numberOfSamples) + "_channel_" + str(len(NNchannels))
    )
    torch.save(model.state_dict(), path)
    if run_dir is not None:
        output_paths = _save_pretraining_outputs(
            run_dir,
            trainingCostHistory,
            validationCostHistory,
            validationGammaMSEHistory,
            validationDiceHistory,
            validationIoUHistory,
        )
        print(f"Saved pretraining histories and plot under {Path(run_dir)}", flush=True)
        for key, value in output_paths.items():
            print(f"{key}: {value}", flush=True)
    return path
