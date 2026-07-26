"""Fast structural checks for the opt-in SegFormer improvement profile."""

import _bootstrap
import torch

from src.experiments.transfer_segformer_fwi import set_segformer_trainable_mode
from src.networks import GradientSegFormer, GradientSegFormerHighResolution
from src.pretrain_segformer import (
    build_pretraining_scheduler,
    build_segformer_model,
    segformer_model_type,
)
from src.segformer_improvements import load_improvement_profile


def trainable_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main():
    config, _ = load_improvement_profile("configs/segformer_improved.yaml")
    model = GradientSegFormer(
        config["models"]["segformer"],
        gamma_min=config["simulation"]["gamma0"],
        void_prior=config["segformer_pretraining"]["void_prior"],
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    decoder = sum(parameter.numel() for parameter in model.segformer.decode_head.parameters())

    selected_decoder = set_segformer_trainable_mode(model, "decoder_only")
    assert selected_decoder == decoder == trainable_count(model)
    decoder_output = model(torch.zeros(1, 1, 256, 128))
    decoder_output.mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in model.segformer.decode_head.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.segformer.segformer.parameters()
    )
    model.zero_grad(set_to_none=True)
    selected_last = set_segformer_trainable_mode(model, "decoder_plus_last_stage")
    assert decoder < selected_last < total
    selected_all = set_segformer_trainable_mode(model, "all")
    assert selected_all == total == trainable_count(model)

    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 1, 256, 128))
    assert output.shape == (1, 1, 256, 128)
    assert torch.isfinite(output).all()

    high_resolution_model, variant = build_segformer_model(
        config,
        {"gamma0": config["simulation"]["gamma0"]},
        model_variant="segformer_highres",
    )
    assert variant == "segformer_highres"
    assert isinstance(
        high_resolution_model,
        GradientSegFormerHighResolution,
    )
    assert (
        segformer_model_type(
            config["segformer_pretraining"],
            variant,
        )
        == "SegFormerHighResolution"
    )
    high_resolution_total = sum(
        parameter.numel()
        for parameter in high_resolution_model.parameters()
    )
    assert high_resolution_total - total == 1393
    high_resolution_model.eval()
    with torch.no_grad():
        gradient = torch.zeros(1, 1, 256, 128)
        coarse_logits = high_resolution_model.forward_coarse_logits(gradient)
        base_logits = high_resolution_model.forward_base_logits(gradient)
        correction = high_resolution_model.forward_correction(
            gradient,
            base_logits=base_logits,
        )
        refined_logits = high_resolution_model.forward_logits(gradient)
    assert coarse_logits.shape == (1, 1, 64, 32)
    assert base_logits.shape == correction.shape == refined_logits.shape
    assert refined_logits.shape == (1, 1, 256, 128)
    assert torch.count_nonzero(correction) == 0
    assert torch.equal(base_logits, refined_logits)

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler, per_step = build_pretraining_scheduler(
        optimizer,
        config["segformer_pretraining"],
        steps_per_epoch=10,
    )
    assert scheduler is not None and per_step
    assert optimizer.param_groups[0]["lr"] < 1e-4

    scaled_config, _ = load_improvement_profile(
        "configs/segformer_improved_15k.yaml"
    )
    assert scaled_config["paths"]["train_data"] == "data/extended"
    assert scaled_config["segformer_pretraining"]["numberOfSamples"] == 15000
    assert scaled_config["models"]["segformer"]["decoder_hidden_size"] == 256

    print(
        "SegFormer improvement checks passed: total={}, high_resolution={}, "
        "decoder={}, decoder_plus_last={}".format(
            total,
            high_resolution_total,
            decoder,
            selected_last,
        )
    )


if __name__ == "__main__":
    main()
