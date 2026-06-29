import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import _bootstrap
from src import adjoint
from src import io
from src import networks as NN
from src.config import load_config
from src.experiments.base import (
    create_forward_solver,
    create_initial_conditions,
    load_case_data,
    simulation_parameters,
)
from src.experiments.transfer_segformer_fwi import save_segformer_layout_plot
from src.pretrain_segformer import dice_loss_from_logits


def has_nonzero_grad(parameters):
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad.detach()).item() > 0
        for parameter in parameters
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--out", default="temp/segformer_smoke")
    args = parser.parse_args()

    config = load_config(args.config)
    params = simulation_parameters(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = io.ensure_dir(args.out)

    model = NN.GradientSegFormer(
        spec=config.get("models", {}).get("segformer", {}),
        gamma_min=params["gamma0"],
        void_prior=config["segformer_pretraining"]["void_prior"],
    ).to(device)
    gradient = torch.randn(2, 1, params["Nx"] + 1, params["Ny"] + 1, device=device)
    norm_cfg = config["segformer_pretraining"]["gradient_normalization"]
    gradient = NN.normalize_gradient_for_transformer(gradient, **norm_cfg)
    logits = model.forward_logits(gradient)
    gamma = model(gradient)
    assert logits.shape == gradient.shape, (logits.shape, gradient.shape)
    assert gamma.shape == gradient.shape, (gamma.shape, gradient.shape)
    assert float(gamma.min()) >= params["gamma0"] - 1e-6
    assert float(gamma.max()) <= 1.0 + 1e-6

    target = (torch.rand_like(logits) > 0.98).float()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = next(model.parameters()).detach().clone()
    loss = criterion(logits, target) + dice_loss_from_logits(logits, target)
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    after = next(model.parameters()).detach()
    assert not torch.equal(before, after)

    with torch.no_grad():
        voidness = model.forward_voidness(gradient[:1])
        gamma_image = model(gradient[:1])
        gamma_solver = torch.ones((1, 1, params["Nx"] + 3, params["Ny"] + 3), device=device)
        gamma_solver[:, :, 1:-1, 1:-1] = gamma_image
    save_segformer_layout_plot(output_dir, "segformer_smoke", args.case, gradient[:1], voidness, gamma_image, gamma_solver)

    if args.online:
        data_dir = Path(config["paths"]["casestudy_data"])
        true_gamma, um, F = load_case_data(args.case, data_dir, params, device, load_gradient=False)
        u0, u1 = create_initial_conditions(params, device)
        solver = create_forward_solver(params, device)
        homogeneous_gamma = torch.ones_like(true_gamma)
        _, first_gradient = adjoint.getAdjointGradient(
            solver,
            u0,
            u1,
            params["c"],
            params["rho"],
            homogeneous_gamma.detach(),
            F,
            params["Nx"],
            params["dx"],
            params["Ny"],
            params["dy"],
            params["N"],
            params["dt"],
            params["numberOfSources"],
            um,
            params["selx"],
            params["sely"],
            device,
        )
        fixed_input = NN.normalize_gradient_for_transformer(
            first_gradient.unsqueeze(0).unsqueeze(0),
            **norm_cfg,
        ).detach()
        gamma_pred = torch.ones_like(true_gamma)
        gamma_pred[:, :, 1:-1, 1:-1] = model(fixed_input)
        cost, external_gradient = adjoint.getAdjointGradient(
            solver,
            u0,
            u1,
            params["c"],
            params["rho"],
            gamma_pred.detach(),
            F,
            params["Nx"],
            params["dx"],
            params["Ny"],
            params["dy"],
            params["N"],
            params["dt"],
            params["numberOfSources"],
            um,
            params["selx"],
            params["sely"],
            device,
        )
        gamma_pred.grad = torch.zeros_like(gamma_pred)
        gamma_pred.grad[0, 0, 1:-1, 1:-1] = external_gradient * 1e10
        optimizer.zero_grad(set_to_none=True)
        gamma_pred.backward(gamma_pred.grad)
        decode_params = [p for name, p in model.named_parameters() if "decode_head" in name]
        encoder_params = [p for name, p in model.named_parameters() if "encoder" in name]
        assert has_nonzero_grad(decode_params)
        assert has_nonzero_grad(encoder_params)
        optimizer.step()
        print("Online one-step cost: {:.6E}".format(float(cost.detach().cpu())))

    plt.close("all")
    print("SegFormer smoke test passed. Outputs saved under {}".format(output_dir))


if __name__ == "__main__":
    main()
