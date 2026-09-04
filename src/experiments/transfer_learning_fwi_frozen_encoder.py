import time
from pathlib import Path

import numpy as np
import torch

from src import adjoint
from src import metrics
from src import networks as NN
from src.experiments.base import (
    ExperimentResult,
    create_forward_solver,
    create_initial_conditions,
    get_device,
    load_case_data,
    model_path,
    normalize_input_data,
    plot_reconstruction_history,
    save_histories,
    save_outputs,
    simulation_parameters,
)

FREEZE_MODES = ("encoder", "decoder", "random_encoder", "none")


def unet_encoder_modules(model):
    """Down path plus bottleneck."""
    return [
        model.convolutionsDown, model.bnormsDown, model.activationsDown,
        model.convolutionsBottleneck, model.bnormsBottleneck,
        model.activationsBottleneck,
    ]


def unet_decoder_modules(model):
    """Up path."""
    return [model.convolutionsUp, model.bnormsUp, model.activationsUp]


def apply_freeze_mode(model, mode, seed=None):
    """
    Freeze part of the U-Net and report what is trainable and what must be held
    in eval mode.

    The literature disagrees about freezing, and the disagreement is confounded
    by domain gap. Yosinski et al. establish general-to-specific and are read as
    implying that freezing early layers should be roughly free; Amiri et al.
    find on ultrasound U-Nets that freezing the encoder and fine-tuning the
    decoder is often the WORST choice -- but they pretrain on natural images, so
    their early layers face a large domain shift. Raghu et al. and Karimi et al.
    argue much of the benefit is initialization scale rather than learned
    features at all.

    This thesis transfers IN-DOMAIN (adjoint gradients -> gamma, same physics
    and generator), which removes the domain-gap confound. The modes below let
    the experiment adjudicate rather than illustrate:

    ``encoder``         freeze the down path + bottleneck, train the decoder.
                        The configuration the field's usual advice implies.
    ``decoder``         the reverse; Amiri et al.'s better-performing direction,
                        tested here without their domain shift.
    ``random_encoder``  RE-INITIALIZE the encoder randomly, freeze it, and train
                        only the decoder. The control for Raghu/Karimi: if this
                        matches a frozen PRETRAINED encoder, then what transfers
                        is not the learned features.
    ``none``            full fine-tuning baseline.

    Frozen BatchNorm modules are returned separately: they must be held in
    eval() every epoch or they keep updating running statistics even with
    requires_grad=False, which silently changes a "frozen" branch.
    """
    if mode not in FREEZE_MODES:
        raise ValueError(f"freeze_mode must be one of {FREEZE_MODES}, got {mode!r}")

    encoder = unet_encoder_modules(model)
    decoder = unet_decoder_modules(model)

    if mode == "none":
        for param in model.parameters():
            param.requires_grad = True
        frozen = []
    else:
        if mode == "random_encoder":
            # Discard the pretrained encoder weights entirely, keeping only the
            # architecture and the initialization scheme.
            if seed is not None:
                torch.manual_seed(int(seed))
            for module in encoder:
                module.apply(NN.initWeights)
        frozen = decoder if mode == "decoder" else encoder
        trainable = encoder if mode == "decoder" else decoder
        for param in model.parameters():
            param.requires_grad = False
        for module in trainable:
            for param in module.parameters():
                param.requires_grad = True

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError(f"freeze_mode {mode!r} left no trainable parameters")
    frozen_batchnorms = [m for m in frozen
                         if isinstance(m, torch.nn.ModuleList)
                         and any(isinstance(x, torch.nn.BatchNorm2d) for x in m)]
    return trainable_parameters, frozen_batchnorms


def freeze_unet_encoder(model):
    """Backwards-compatible alias for the original encoder-freezing behaviour."""
    return apply_freeze_mode(model, "encoder")[0]



def set_optimization_model_mode(model, mode):
    """
    Whether the network runs in train() or eval() mode during the inversion.

    This matters more than it looks. FWI optimizes ONE case, so every forward
    pass sees a batch of exactly one image. In train() mode BatchNorm then
    normalizes each channel by that single image's own spatial statistics --
    which is InstanceNorm, not the population statistics estimated from batches
    of 80 during pretraining. The network is therefore evaluated under a
    different normalization than it was fitted with.

    "train" reproduces legacy/TransferLearningFWI.py exactly and is the default.
    "eval" uses the stored running statistics, matching how transfer_segformer_fwi
    has always been run (`optimization_model_mode: eval`). Autograd is unaffected
    either way, so the weights still train in both modes.
    """
    if mode == "train":
        model.train()
    elif mode == "eval":
        model.eval()
    else:
        raise ValueError("optimization_model_mode must be train or eval")

class TransferLearningFWIFrozenEncoder:
    name = "transfer_learning_fwi_frozen_encoder"

    def __init__(self, config, device=None):
        self.config = config
        self.device = get_device(device)
        self.params = simulation_parameters(config)

    def run(self, case_id: int, data_dir: Path, run_dir: Path):
        params = self.params
        cfg = self.config["experiments"][self.name]
        exp_cfg = self.config["experiments"]
        device = self.device
        print("Using device: {}".format(device))
        print("Number of sensors: {:d}".format(len(params["selx"])))
        print(case_id)

        sample = int(exp_cfg["pretrain_samples"][0])
        NNchannels = exp_cfg["NNchannels"]
        numberOfConvolutionsPerBlock = int(exp_cfg["numberOfConvolutionsPerBlock"])
        modelType = exp_cfg["modelType"]
        epochs_pretrain = int(exp_cfg["epochs_pretrain"])
        path = model_path(
            self.config,
            modelType,
            epochs_pretrain,
            "supervised",
            sample,
            NNchannels,
        )

        gamma, initialGradient, um, F = load_case_data(
            case_id, data_dir, params, device, load_gradient=True
        )
        u0, u1 = create_initial_conditions(params, device)

        torch.manual_seed(int(cfg["seed"]))
        model = NN.Unet(
            NNchannels, numberOfConvolutionsPerBlock, params["gamma0"], bnorm=True
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)

        optimization_model_mode = str(cfg.get("optimization_model_mode", "train"))
        freeze_mode = str(cfg.get("freeze_mode", "encoder"))
        trainable_parameters, frozen_batchnorms = apply_freeze_mode(
            model, freeze_mode, seed=cfg.get("seed")
        )
        total_parameters = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in trainable_parameters)
        print(
            "freeze_mode={}: {:,}/{:,} trainable ({:.1f}%)".format(
                freeze_mode, trainable_count, total_parameters,
                100.0 * trainable_count / max(1, total_parameters),
            ),
            flush=True,
        )
        forwardSolver = create_forward_solver(params, device)
        inputData = normalize_input_data(initialGradient, self.config).to(device)

        gammaPred = torch.ones(
            (1, 1, params["Nx"] + 3, params["Ny"] + 3),
            device=device,
            dtype=torch.float32,
        )
        gammaPred[:, :, 1:-1, 1:-1] = model(inputData)

        epochs = int(cfg["epochs"])
        optimizer = torch.optim.Adam(trainable_parameters, cfg["lr"])
        lr_lambda = lambda epoch: (cfg["beta"] * epoch + 1) ** cfg["alpha"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        costHistory = np.zeros(epochs)
        mseHistory = np.zeros(epochs)
        gammaHistory = np.zeros((epochs + 1, params["Nx"] + 1, params["Ny"] + 1))
        gammaHistory[0] = gammaPred[0, 0, 1:-1, 1:-1].cpu().detach().numpy()

        start = time.perf_counter()
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            set_optimization_model_mode(model, optimization_model_mode)

            # Frozen BatchNorms must be held in eval(): requires_grad=False does
            # NOT stop them updating running statistics.
            for module in frozen_batchnorms:
                module.eval()
            #----------------------------------------------
            
            gammaPred = torch.ones(
                (1, 1, params["Nx"] + 3, params["Ny"] + 3),
                device=device,
                dtype=torch.float32,
            )
            gammaPred[:, :, 1:-1, 1:-1] = model(inputData)
            gammaPred.grad = torch.zeros_like(gammaPred.detach(), device=device)

            cost, gradient = adjoint.getAdjointGradient(
                forwardSolver,
                u0,
                u1,
                params["c"],
                params["rho"],
                gammaPred.detach(),
                F,
                params["Nx"],
                params["dx"],
                params["Ny"],
                params["dy"],
                params["N"],
                params["dt"],
                params["numberOfSources"],
                um.to(device),
                params["selx"],
                params["sely"],
                device,
            )
            gammaPred.grad[0, 0, 1:-1, 1:-1] = gradient * cfg["costScaling"]
            gammaPred.backward(gammaPred.grad)

            torch.nn.utils.clip_grad_norm_(trainable_parameters, cfg["clipGrad"])
            optimizer.step()
            scheduler.step()

            costHistory[epoch] = cost.detach().cpu()
            mseHistory[epoch] = metrics.gamma_mse(gammaPred, gamma, ghost=1)
            gammaHistory[epoch + 1] = gammaPred[0, 0, 1:-1, 1:-1].detach().cpu()

            elapsed_time = time.perf_counter() - start
            if epoch % 2 == 0:
                string = "Epoch: {}/{}\t\tCost function: {:.7E}\t\tMSE: {:.7E}\t\tElapsed time: {:2f}"
                print(string.format(epoch, epochs - 1, costHistory[epoch], mseHistory[epoch], elapsed_time))
            start = time.perf_counter()

        plot_reconstruction_history(
            gammaHistory,
            gamma,
            epochs,
            Path(run_dir) / "figures" / f"{self.name}_case{case_id}_figure.svg",
            mse_history=mseHistory,
        )
        save_histories(
            run_dir,
            f"{self.name}_case{case_id}_gamma_history",
            f"{self.name}_case{case_id}_cost_history",
            f"{self.name}_case{case_id}_mse_history",
            gammaHistory,
            costHistory,
            mseHistory,
        )
        target_gamma = gamma[0, 0, 1:-1, 1:-1].detach().cpu().numpy()
        save_outputs(run_dir, self.name, case_id, gammaHistory[-1], target_gamma)

        return ExperimentResult(
            method_name=self.name,
            case_id=case_id,
            gamma_history=gammaHistory,
            cost_history=costHistory,
            mse_history=mseHistory,
            final_gamma=gammaHistory[-1],
            target_gamma=target_gamma,
            run_dir=Path(run_dir),
            metadata={
                "epochs": epochs,
                "freeze_mode": freeze_mode,
                "trainable_parameters": trainable_count,
                "total_parameters": total_parameters,
                "pretrain_samples": sample,
                "epochs_pretrain": epochs_pretrain,
            },
        )
