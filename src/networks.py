from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


# dataset definition
class FWIDataset(Dataset):
    def __init__(self, x, y, device, u=None):
        self.x = x
        self.y = y
        self.device = device
        self.u = u

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.u == None:
            return self.x[idx].to(self.device), self.y[idx].to(self.device)
        else:
            return (
                self.x[idx].to(self.device),
                self.y[idx].to(self.device),
                self.u[idx].to(self.device),
            )


class makeAdaptiveActivation(torch.nn.Module):
    def __init__(self, n, activation):
        super().__init__()
        self.n = n
        self.alpha = torch.nn.parameter.Parameter(torch.tensor(1.0 / n))
        self.activation = activation

    def forward(self, x):
        return self.activation(self.n * self.alpha * x)


def initWeights(m):
    """Initialize weights of neural network with xavier initialization."""
    if (
            type(m) == torch.nn.Linear
            or type(m) == torch.nn.Conv2d
            or type(m) == torch.nn.Conv3d
    ):
        torch.nn.init.xavier_uniform_(
            m.weight, gain=torch.nn.init.calculate_gain("leaky_relu", 0.2))  # xavier somehow performs better than He
        m.bias.data.fill_(0.0)
    elif type(m) == torch.nn.PReLU:
        m.weight.data.fill_(0.2)


def PixelNorm(x):
    return x / torch.sqrt(
        torch.sum(x ** 2, axis=(2, 3), keepdim=True) / x.shape[2] / x.shape[3] + 1e-8
    )


@dataclass
class SegFormerSpec:
    in_channels: int = 1
    hidden_sizes: tuple = (24, 48, 96, 192)
    depths: tuple = (2, 2, 2, 2)
    num_attention_heads: tuple = (1, 2, 4, 8)
    sr_ratios: tuple = (8, 4, 2, 1)
    patch_sizes: tuple = (7, 3, 3, 3)
    strides: tuple = (4, 2, 2, 2)
    mlp_ratios: tuple = (4, 4, 4, 4)
    # Backward-compatible fallback. Final experiments explicitly override this
    # with decoder_hidden_size: 256 in configs/extended.yaml.
    decoder_hidden_size: int = 64
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    classifier_dropout_prob: float = 0.0
    drop_path_rate: float = 0.0
    decoder_norm: str = "batch"
    decoder_norm_groups: int = 8

    @classmethod
    def from_dict(cls, values):
        if values is None:
            return cls()
        tuple_keys = {
            "hidden_sizes",
            "depths",
            "num_attention_heads",
            "sr_ratios",
            "patch_sizes",
            "strides",
            "mlp_ratios",
        }
        cleaned = {
            key: tuple(value) if key in tuple_keys else value
            for key, value in values.items()
            if key in cls.__dataclass_fields__
        }
        return cls(**cleaned)

    def to_dict(self):
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, tuple):
                values[key] = list(value)
        return values

    def validate(self):
        sequence_lengths = {
            len(self.hidden_sizes),
            len(self.depths),
            len(self.num_attention_heads),
            len(self.sr_ratios),
            len(self.patch_sizes),
            len(self.strides),
            len(self.mlp_ratios),
        }
        if sequence_lengths != {4}:
            raise ValueError("SegFormerSpec expects four stages for all stage-wise settings")
        for stage, (hidden_size, heads) in enumerate(
            zip(self.hidden_sizes, self.num_attention_heads)
        ):
            if hidden_size % heads != 0:
                raise ValueError(
                    "SegFormer hidden size must be divisible by attention heads "
                    f"at stage {stage}: {hidden_size} % {heads} != 0"
                )
        if self.decoder_norm not in ("batch", "batch_no_running", "group"):
            raise ValueError(
                "SegFormer decoder_norm must be batch, batch_no_running, or group"
            )
        if self.decoder_norm == "group":
            groups = int(self.decoder_norm_groups)
            if groups < 1 or self.decoder_hidden_size % groups != 0:
                raise ValueError(
                    "decoder_hidden_size must be divisible by decoder_norm_groups"
                )


def normalize_gradient(
    gradient,
    mode="robust_abs",
    quantile=0.99,
    eps=1e-8,
    clamp=1.0,
):
    """
    Shared adjoint-gradient normalization, used by EVERY model.

    Two modes:

    ``robust_abs`` (default)
        Sign-preserving division by a high quantile of |gradient|, then a
        symmetric clamp. Preferred because an adjoint gradient carries large
        isolated spikes at the source positions; a scale set by those spikes
        compresses the informative bulk of the field into a narrow band.

    ``minmax``
        Per-sample affine map of [min, max] onto [-1, 1]. This is the legacy
        U-Net convention, retained so old behaviour can be reproduced. It is
        defined by exactly two extreme pixels and is therefore sensitive to the
        source spikes described above.

    Using one function for both architectures is what allows the U-Net vs
    SegFormer comparison to isolate architecture rather than preprocessing.
    """
    squeeze_channel = False
    if gradient.ndim == 3:
        gradient = gradient.unsqueeze(1)
        squeeze_channel = True
    if gradient.ndim != 4:
        raise ValueError(f"gradient must have shape [B, 1, H, W], got {tuple(gradient.shape)}")

    if mode in (None, "none"):
        normalized = gradient
    elif mode == "robust_abs":
        flat_abs = gradient.detach().abs().flatten(start_dim=1)
        scale = torch.quantile(flat_abs, float(quantile), dim=1)
        scale = torch.clamp(scale, min=float(eps)).view(-1, 1, 1, 1)
        normalized = gradient / scale
        if clamp is not None:
            normalized = torch.clamp(normalized, -float(clamp), float(clamp))
    elif mode == "minmax":
        low = torch.amin(gradient, (2, 3), keepdim=True)
        high = torch.amax(gradient, (2, 3), keepdim=True)
        normalized = (gradient - low) / torch.clamp(high - low, min=float(eps)) * 2 - 1
    else:
        raise ValueError(f"Unknown gradient normalization mode: {mode}")

    if squeeze_channel:
        return normalized.squeeze(1)
    return normalized


# Backwards-compatible alias: this used to be SegFormer-specific, but the same
# normalization is now shared by every architecture.
normalize_gradient_for_transformer = normalize_gradient


class GradientSegFormer(torch.nn.Module):
    """
    SegFormer wrapper mapping one adjoint-gradient image to differentiable gamma.

    Input:  [B, 1, H, W]
    Logits: [B, 1, H, W]
    Gamma:  [B, 1, H, W]
    """

    def __init__(self, spec=None, gamma_min=1.0e-5, void_prior=0.01):
        super().__init__()
        self.spec = SegFormerSpec.from_dict(spec) if isinstance(spec, dict) else (spec or SegFormerSpec())
        self.spec.validate()
        self.void_prior = float(void_prior)

        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "GradientSegFormer requires the 'transformers' package. "
                "Install project dependencies or run: pip install transformers"
            ) from exc

        config = SegformerConfig(
            num_channels=int(self.spec.in_channels),
            num_labels=1,
            hidden_sizes=list(self.spec.hidden_sizes),
            depths=list(self.spec.depths),
            num_attention_heads=list(self.spec.num_attention_heads),
            sr_ratios=list(self.spec.sr_ratios),
            patch_sizes=list(self.spec.patch_sizes),
            strides=list(self.spec.strides),
            mlp_ratios=list(self.spec.mlp_ratios),
            decoder_hidden_size=int(self.spec.decoder_hidden_size),
            hidden_dropout_prob=float(self.spec.hidden_dropout_prob),
            attention_probs_dropout_prob=float(self.spec.attention_probs_dropout_prob),
            classifier_dropout_prob=float(self.spec.classifier_dropout_prob),
            drop_path_rate=float(self.spec.drop_path_rate),
        )
        self.segformer = SegformerForSemanticSegmentation(config)
        if self.spec.decoder_norm == "batch_no_running":
            self.segformer.decode_head.batch_norm = torch.nn.BatchNorm2d(
                int(self.spec.decoder_hidden_size),
                track_running_stats=False,
            )
        elif self.spec.decoder_norm == "group":
            self.segformer.decode_head.batch_norm = torch.nn.GroupNorm(
                num_groups=int(self.spec.decoder_norm_groups),
                num_channels=int(self.spec.decoder_hidden_size),
            )
        self.register_buffer("gamma_min", torch.tensor(float(gamma_min), dtype=torch.float32))
        self.reset_classifier_bias()

    def reset_classifier_bias(self):
        if not (0.0 < self.void_prior < 1.0):
            raise ValueError(f"void_prior must be in (0, 1), got {self.void_prior}")
        prior_logit = math.log(self.void_prior / (1.0 - self.void_prior))
        classifier = getattr(self.segformer.decode_head, "classifier", None)
        if classifier is None or classifier.bias is None:
            raise RuntimeError("Could not find SegFormer decode-head classifier bias")
        with torch.no_grad():
            classifier.bias.fill_(prior_logit)

    def architecture_dict(self):
        return self.spec.to_dict()

    def forward_logits(self, gradient_input):
        if gradient_input.ndim != 4 or gradient_input.shape[1] != self.spec.in_channels:
            raise ValueError(
                "GradientSegFormer expects input shape "
                f"[B, {self.spec.in_channels}, H, W], got {tuple(gradient_input.shape)}"
            )
        output = self.segformer(pixel_values=gradient_input)
        logits = output.logits
        if logits.shape[-2:] != gradient_input.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=gradient_input.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def forward_voidness(self, gradient_input):
        return torch.sigmoid(self.forward_logits(gradient_input))

    def forward(self, gradient_input):
        p_void = self.forward_voidness(gradient_input)
        return 1.0 - (1.0 - self.gamma_min) * p_void


class GradientSegFormerHighResolution(torch.nn.Module):
    """
    SegFormer with a shallow full-resolution residual refinement branch.

    The original SegFormer head predicts coarse logits at one-quarter input
    resolution. This class keeps that architecture intact, upsamples its logits,
    extracts local features directly from the full-resolution gradient, and
    predicts a residual logit correction:

        refined_logits = upsampled_segformer_logits + correction

    Input:      [B, in_channels, H, W]
    Coarse:     [B, 1, H/4, W/4]
    Correction: [B, 1, H, W]
    Logits:     [B, 1, H, W]
    Gamma:      [B, 1, H, W]
    """

    def __init__(
        self,
        spec=None,
        gamma_min=1.0e-5,
        void_prior=0.01,
        refiner_channels=8,
    ):
        super().__init__()
        if isinstance(spec, dict):
            self.spec = SegFormerSpec.from_dict(spec)
            refiner_channels = spec.get("refiner_channels", refiner_channels)
        else:
            self.spec = spec or SegFormerSpec()
        self.spec.validate()

        self.void_prior = float(void_prior)
        self.refiner_channels = int(refiner_channels)
        if self.refiner_channels < 1:
            raise ValueError(
                f"refiner_channels must be positive, got {self.refiner_channels}"
            )

        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "GradientSegFormerHighResolution requires the 'transformers' "
                "package. Install project dependencies or run: "
                "pip install transformers"
            ) from exc

        config = SegformerConfig(
            num_channels=int(self.spec.in_channels),
            num_labels=1,
            hidden_sizes=list(self.spec.hidden_sizes),
            depths=list(self.spec.depths),
            num_attention_heads=list(self.spec.num_attention_heads),
            sr_ratios=list(self.spec.sr_ratios),
            patch_sizes=list(self.spec.patch_sizes),
            strides=list(self.spec.strides),
            mlp_ratios=list(self.spec.mlp_ratios),
            decoder_hidden_size=int(self.spec.decoder_hidden_size),
            hidden_dropout_prob=float(self.spec.hidden_dropout_prob),
            attention_probs_dropout_prob=float(
                self.spec.attention_probs_dropout_prob
            ),
            classifier_dropout_prob=float(self.spec.classifier_dropout_prob),
            drop_path_rate=float(self.spec.drop_path_rate),
        )
        self.segformer = SegformerForSemanticSegmentation(config)
        if self.spec.decoder_norm == "batch_no_running":
            self.segformer.decode_head.batch_norm = torch.nn.BatchNorm2d(
                int(self.spec.decoder_hidden_size),
                track_running_stats=False,
            )
        elif self.spec.decoder_norm == "group":
            self.segformer.decode_head.batch_norm = torch.nn.GroupNorm(
                num_groups=int(self.spec.decoder_norm_groups),
                num_channels=int(self.spec.decoder_hidden_size),
            )

        channels = self.refiner_channels
        self.high_resolution_input = torch.nn.Sequential(
            torch.nn.Conv2d(
                int(self.spec.in_channels),
                channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.GELU(),
            torch.nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.GELU(),
        )
        self.high_resolution_fusion = torch.nn.Sequential(
            torch.nn.Conv2d(
                channels + 1,
                channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.GELU(),
        )
        self.high_resolution_correction = torch.nn.Conv2d(
            channels,
            1,
            kernel_size=3,
            padding=1,
        )

        self.register_buffer(
            "gamma_min",
            torch.tensor(float(gamma_min), dtype=torch.float32),
        )
        self.reset_classifier_bias()
        self.reset_refiner_correction()

    def reset_classifier_bias(self):
        if not (0.0 < self.void_prior < 1.0):
            raise ValueError(
                f"void_prior must be in (0, 1), got {self.void_prior}"
            )
        prior_logit = math.log(self.void_prior / (1.0 - self.void_prior))
        classifier = getattr(self.segformer.decode_head, "classifier", None)
        if classifier is None or classifier.bias is None:
            raise RuntimeError("Could not find SegFormer decode-head classifier bias")
        with torch.no_grad():
            classifier.bias.fill_(prior_logit)

    def reset_refiner_correction(self):
        """Start as the unmodified SegFormer by predicting zero correction."""
        torch.nn.init.zeros_(self.high_resolution_correction.weight)
        if self.high_resolution_correction.bias is not None:
            torch.nn.init.zeros_(self.high_resolution_correction.bias)

    def architecture_dict(self):
        architecture = self.spec.to_dict()
        architecture.update(
            {
                "high_resolution_refiner": True,
                "refiner_channels": self.refiner_channels,
                "residual_logit_correction": True,
            }
        )
        return architecture

    def _validate_gradient_input(self, gradient_input):
        if (
            gradient_input.ndim != 4
            or gradient_input.shape[1] != self.spec.in_channels
        ):
            raise ValueError(
                "GradientSegFormerHighResolution expects input shape "
                f"[B, {self.spec.in_channels}, H, W], "
                f"got {tuple(gradient_input.shape)}"
            )

    def forward_coarse_logits(self, gradient_input):
        """Return the original SegFormer decoder logits before final resizing."""
        self._validate_gradient_input(gradient_input)
        return self.segformer(pixel_values=gradient_input).logits

    def forward_base_logits(self, gradient_input):
        """Return the original SegFormer logits resized to the input grid."""
        coarse_logits = self.forward_coarse_logits(gradient_input)
        if coarse_logits.shape[-2:] == gradient_input.shape[-2:]:
            return coarse_logits
        return F.interpolate(
            coarse_logits,
            size=gradient_input.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward_correction(self, gradient_input, base_logits=None):
        """Predict a full-resolution residual correction to the base logits."""
        self._validate_gradient_input(gradient_input)
        if base_logits is None:
            base_logits = self.forward_base_logits(gradient_input)
        local_features = self.high_resolution_input(gradient_input)
        combined = torch.cat((base_logits, local_features), dim=1)
        fused_features = self.high_resolution_fusion(combined)
        return self.high_resolution_correction(fused_features)

    def forward_logits(self, gradient_input):
        base_logits = self.forward_base_logits(gradient_input)
        correction = self.forward_correction(
            gradient_input,
            base_logits=base_logits,
        )
        return base_logits + correction

    def forward_voidness(self, gradient_input):
        return torch.sigmoid(self.forward_logits(gradient_input))

    def forward(self, gradient_input):
        p_void = self.forward_voidness(gradient_input)
        return 1.0 - (1.0 - self.gamma_min) * p_void


# NN definition
class Unet(torch.nn.Module):
    def __init__(self, channels, numberOfConvolutionsPerBlock, gamma0, bnorm=True):
        super().__init__()

        self.channels = channels
        self.numberOfConvolutionsPerBlock = numberOfConvolutionsPerBlock
        self.gamma0 = gamma0
        self.bnorm = bnorm

        self.convolutionsDown = torch.nn.ModuleList()
        self.bnormsDown = torch.nn.ModuleList()
        self.activationsDown = torch.nn.ModuleList()
        self.convolutionsBottleneck = torch.nn.ModuleList()
        self.bnormsBottleneck = torch.nn.ModuleList()
        self.activationsBottleneck = torch.nn.ModuleList()
        self.convolutionsUp = torch.nn.ModuleList()
        self.bnormsUp = torch.nn.ModuleList()
        self.activationsUp = torch.nn.ModuleList()

        for i in range(len(self.channels) - 1):
            self.convolutionsDown.append(
                torch.nn.Conv2d(
                    self.channels[i],
                    self.channels[i + 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            self.bnormsDown.append(torch.nn.BatchNorm2d(self.channels[i + 1]))
            self.activationsDown.append(
                torch.nn.PReLU(init=0.2)
            )
            for j in range(self.numberOfConvolutionsPerBlock - 1):
                self.convolutionsDown.append(
                    torch.nn.Conv2d(
                        self.channels[i + 1],
                        self.channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                self.bnormsDown.append(torch.nn.BatchNorm2d(self.channels[i + 1]))
                self.activationsDown.append(torch.nn.PReLU(init=0.2))

        for j in range(self.numberOfConvolutionsPerBlock):
            self.convolutionsBottleneck.append(
                torch.nn.Conv2d(
                    self.channels[-1],
                    self.channels[-1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            self.bnormsBottleneck.append(torch.nn.BatchNorm2d(self.channels[-1]))
            self.activationsBottleneck.append(torch.nn.PReLU(init=0.2))

        for i in range(1, len(self.channels)):
            self.convolutionsUp.append(
                torch.nn.Conv2d(
                    self.channels[-i] + self.channels[-(i + 1)],
                    self.channels[-(i + 1)],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            self.bnormsUp.append(torch.nn.BatchNorm2d(self.channels[-(i + 1)]))
            self.activationsUp.append(torch.nn.PReLU(init=0.2))
            for j in range(self.numberOfConvolutionsPerBlock - 1):
                self.convolutionsUp.append(
                    torch.nn.Conv2d(
                        self.channels[-(i + 1)],
                        self.channels[-(i + 1)],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                self.bnormsUp.append(torch.nn.BatchNorm2d(self.channels[-(i + 1)]))
                if (i < len(self.channels) - 1) and (
                        j < self.numberOfConvolutionsPerBlock - 1
                ):
                    self.activationsUp.append(torch.nn.PReLU(init=0.2))
                else:
                    self.activationsUp.append(
                        torch.nn.Sigmoid()
                    )

        self.downsample = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        torch.use_deterministic_algorithms(True)
        x_ = []
        for i in range(len(self.channels) - 1):
            x_.append(x)
            for j in range(self.numberOfConvolutionsPerBlock):
                x = self.convolutionsDown[self.numberOfConvolutionsPerBlock * i + j](x)
                if self.bnorm == True:
                    x = self.bnormsDown[self.numberOfConvolutionsPerBlock * i + j](x)
                else:
                    x = PixelNorm(x)
                x = self.activationsDown[self.numberOfConvolutionsPerBlock * i + j](x)
            x = self.downsample(x)

        for j in range(self.numberOfConvolutionsPerBlock):
            x = self.convolutionsBottleneck[j](x)
            if self.bnorm == True:
                x = self.bnormsBottleneck[j](x)
            else:
                x = PixelNorm(x)
            x = self.activationsBottleneck[j](x)
        # Decoder. The reference implementation (legacy/NeuralNetwork.py:174-184)
        # normalized only the FIRST convolution of each up-block, while the
        # encoder normalizes after every convolution. That left half of the
        # `bnormsUp` modules constructed but never called -- dead parameters
        # carried in every checkpoint. Normalization is now applied symmetrically
        # with the encoder, with ONE deliberate exception: the final output
        # convolution is left unnormalized, because its weights and bias are
        # initialized (mean 0.7, bias 3) specifically to put the following
        # Sigmoid deep in its saturated region so that gamma starts at intact
        # material. A BatchNorm there would re-centre the pre-activation to zero
        # mean and destroy that prior.
        last_up = len(self.convolutionsUp) - 1
        for i in range(len(self.channels) - 1):
            x = self.upsample(x)
            index = self.numberOfConvolutionsPerBlock * i
            x = self.convolutionsUp[index](torch.cat((x, x_[-(i + 1)]), 1))
            if self.bnorm == True and index != last_up:
                x = self.bnormsUp[index](x)
            x = self.activationsUp[index](x)
            for j in range(1, self.numberOfConvolutionsPerBlock):
                index = self.numberOfConvolutionsPerBlock * i + j
                x = self.convolutionsUp[index](x)
                if self.bnorm == True and index != last_up:
                    x = self.bnormsUp[index](x)
                x = self.activationsUp[index](x)

        x = x * (1 - self.gamma0) + self.gamma0

        return x


class CNNGamma(torch.nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

        n = 10

        self.upsample = torch.nn.Upsample(scale_factor=2,
                                          mode='nearest')  # nearest instead of bilinear as field is not continuous

        self.conv1 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.activation1 = torch.nn.PReLU(init=0.2)
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.activation2 = torch.nn.PReLU(init=0.2)

        self.conv3 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.activation3 = torch.nn.PReLU(init=0.2)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.activation4 = torch.nn.PReLU(init=0.2)

        self.conv5 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.activation5 = torch.nn.PReLU(init=0.2)
        self.conv6 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.activation6 = torch.nn.PReLU(init=0.2)

        self.conv7 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.activation7 = torch.nn.PReLU(init=0.2)
        self.conv8 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.activation8 = torch.nn.PReLU(init=0.2)

        self.conv9 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.activation9 = torch.nn.PReLU(init=0.2)
        self.conv10 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.activation10 = torch.nn.PReLU(init=0.2)

        self.convOut = torch.nn.Conv2d(32, 1, kernel_size=3, stride=1)
        self.activationOut = makeAdaptiveActivation(n, torch.nn.Sigmoid())

    def forward(self, x):
        x = self.upsample(x)
        x = PixelNorm(self.activation1.forward(self.conv1(x)))
        x = PixelNorm(self.activation2.forward(self.conv2(x)))

        x = self.upsample(x)
        x = PixelNorm(self.activation3.forward(self.conv3(x)))
        x = PixelNorm(self.activation4.forward(self.conv4(x)))

        x = self.upsample(x)
        x = PixelNorm(self.activation5.forward(self.conv5(x)))
        x = PixelNorm(self.activation6.forward(self.conv6(x)))

        x = self.upsample(x)
        x = PixelNorm(self.activation7.forward(self.conv7(x)))
        x = PixelNorm(self.activation8.forward(self.conv8(x)))

        x = self.upsample(x)
        x = PixelNorm(self.activation9.forward(self.conv9(x)))
        x = PixelNorm(self.activation10.forward(self.conv10(x)))

        x = self.activationOut.forward(self.convOut(x))

        x = (1 - self.epsilon) * x + self.epsilon  # scaling to indicator function

        x = x[0, 0]

        return x


class INR(torch.nn.Module):
    def __init__(
        self,
        hidden_features,
        hidden_layers,
        gamma0,
        output_mode="direct_gamma",
        final_bias=3.0,
    ):
        super().__init__()
        self.gamma0 = gamma0
        self.output_mode = output_mode
        self.final_bias = final_bias
        layers = []
        in_features = 2
        for _ in range(hidden_layers):
            layers.append(torch.nn.Linear(in_features, hidden_features))
            layers.append(torch.nn.Tanh())
            in_features = hidden_features
        self.features = torch.nn.Sequential(*layers)
        self.final_layer = torch.nn.Linear(in_features, 1)
        self.reset_output_bias()

    def reset_output_bias(self):
        with torch.no_grad():
            self.final_layer.bias.fill_(self.final_bias)

    def gamma_from_raw(self, raw):
        """
        Map raw network output to the indicator gamma in [gamma0, 1].

        The two modes are the SAME MODEL: since sigmoid(-r) = 1 - sigmoid(r),

            voidness(raw) == direct_gamma(-raw)

        exactly (verified numerically to float precision). They differ only by
        negating the final layer's weights and bias, so `voidness` adds no
        capacity. `direct_gamma` is the default because it is the convention
        published by Singh et al. (Comput. Mech. 76, 2025) and Herrmann et al.
        (CMAME 415, 2023), where a positive final bias (+3) makes the network
        start at intact material. `voidness` is retained only so that older
        checkpoints remain loadable.
        """
        if self.output_mode == "direct_gamma":
            return self.gamma0 + (1.0 - self.gamma0) * torch.sigmoid(raw)
        if self.output_mode == "voidness":
            p_void = torch.sigmoid(raw)
            return 1.0 - (1.0 - self.gamma0) * p_void
        raise ValueError(f"Unknown INR output_mode: {self.output_mode}")

    def forward(self, coords):
        raw = self.final_layer(self.features(coords))
        return self.gamma_from_raw(raw)


class INRSIREN(torch.nn.Module):
    def __init__(
        self,
        hidden_features,
        hidden_layers,
        gamma0,
        omega0=30,
        output_mode="direct_gamma",
        final_bias=3.0,
    ):
        super().__init__()
        self.gamma0 = gamma0
        self.omega0 = omega0
        self.output_mode = output_mode
        self.final_bias = final_bias
        self.layers = torch.nn.ModuleList()

        in_features = 2
        for _ in range(hidden_layers):
            self.layers.append(torch.nn.Linear(in_features, hidden_features))
            in_features = hidden_features
        self.layers.append(torch.nn.Linear(in_features, 1))
        self.init_siren_weights()

    def init_siren_weights(self):
        with torch.no_grad():
            # SIREN/IFWI initialization: the first layer is not divided by omega0.
            # The omega0 factor is applied to the first layer output in forward().
            first_fan_in = self.layers[0].weight.shape[1]
            self.layers[0].weight.uniform_(-1 / first_fan_in, 1 / first_fan_in)
            for layer in self.layers[1:]:
                fan_in = layer.weight.shape[1]
                bound = np.sqrt(6 / fan_in) / self.omega0
                layer.weight.uniform_(-bound, bound)
            self.layers[-1].bias.fill_(self.final_bias)

    def gamma_from_raw(self, raw):
        if self.output_mode == "voidness":
            # The SIREN predicts voidness logits, not gamma directly.
            # p_void = sigmoid(raw), and gamma = 1 - (1 - gamma_min) * p_void.
            # A negative final bias makes p_void small and initializes gamma near 1.
            p_void = torch.sigmoid(raw)
            return 1.0 - (1.0 - self.gamma0) * p_void
        if self.output_mode == "direct_gamma":
            return self.gamma0 + (1.0 - self.gamma0) * torch.sigmoid(raw)
        raise ValueError(f"Unknown INR output_mode: {self.output_mode}")

    def forward(self, coords):
        x = coords
        for layer in self.layers[:-1]:
            x = torch.sin(self.omega0 * layer(x))
        raw = self.layers[-1](x)
        return self.gamma_from_raw(raw)


class INRSIREN_CENTERED(INRSIREN):
    """
    SIREN-INR with centered residual logits.
    Difference from INRSIREN: raw = final_bias + (final_layer(features) - mean(final_layer(features))).
    """

    def __init__(self, *args, **kwargs):
        final_bias = kwargs.get("final_bias", 3.0)
        if len(args) >= 6:
            final_bias = args[5]
        super().__init__(*args, **kwargs)
        del self.final_bias
        self.register_buffer("final_bias", torch.tensor(float(final_bias)))

        with torch.no_grad():
            self.layers[-1].bias.zero_()

    def forward(self, coords):
        x = coords
        for layer in self.layers[:-1]:
            x = torch.sin(self.omega0 * layer(x))
        residual = self.layers[-1](x)
        residual = residual - residual.mean(dim=0, keepdim=True)
        raw = self.final_bias + residual
        return self.gamma_from_raw(raw)
    
    
class INR_LR(torch.nn.Module):
    """
    Low-rank INR material-field parametrization for LR-FWI.
    this model uses two axis-wise SIREN networks:
        x -> F_x(x) in R^rank_x
        y -> F_y(y) in R^rank_y
    and a trainable core matrix C of shape [rank_x, rank_y]:
        raw(x, y) = F_x(x)^T * C * F_y(y) + final_bias

    The two ranks are independent, following Chen et al. (LR-IFWI, IEEE TGRS
    2025), who set r1 = 50 and r2 = 100 throughout their experiments, and Chen
    et al. (ICLR 2026) App. B.1, which sets the rank to "half of the model
    dimension" -- both of which give a RECTANGULAR core on a non-square domain.
    Passing a single `rank` keeps the old square behaviour.
    """

    def __init__(
        self,
        rank=None,
        hidden_features=128,
        hidden_layers=3,
        gamma0=1e-5,
        omega0=30,
        output_mode="direct_gamma",
        final_bias=3.0,
        core_init_std=1e-3,
        rank_x=None,
        rank_y=None,
    ):
        super().__init__()

        if rank_x is None or rank_y is None:
            if rank is None:
                raise ValueError("INR_LR needs either `rank` or both `rank_x` and `rank_y`")
            rank_x = int(rank_x if rank_x is not None else rank)
            rank_y = int(rank_y if rank_y is not None else rank)
        self.rank_x = int(rank_x)
        self.rank_y = int(rank_y)
        # Kept for checkpoint/back-compat reporting; only meaningful when square.
        self.rank = self.rank_x
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.gamma0 = gamma0
        self.omega0 = omega0
        self.output_mode = output_mode
        self.final_bias = final_bias

        self.x_layers = torch.nn.ModuleList()
        self.y_layers = torch.nn.ModuleList()

        # x network: input dimension 1
        in_features = 1
        for _ in range(hidden_layers):
            self.x_layers.append(torch.nn.Linear(in_features, hidden_features))
            in_features = hidden_features
        self.x_layers.append(torch.nn.Linear(in_features, self.rank_x))

        # y network: input dimension 1
        in_features = 1
        for _ in range(hidden_layers):
            self.y_layers.append(torch.nn.Linear(in_features, hidden_features))
            in_features = hidden_features
        self.y_layers.append(torch.nn.Linear(in_features, self.rank_y))

        # Trainable low-rank core matrix C, [rank_x, rank_y].
        self.core = torch.nn.Parameter(torch.empty(self.rank_x, self.rank_y))

        # Trainable scalar bias added to the raw logits, so that gamma starts
        # at intact material.
        self.raw_bias = torch.nn.Parameter(torch.tensor(float(final_bias)))

        self.init_siren_weights(core_init_std=core_init_std)

    def init_siren_stack(self, layers):
        """
        SIREN initialization for the axis networks.

        IMPORTANT: SIREN's sqrt(6/fan_in)/omega0 bound is DERIVED for layers
        whose output is consumed by sin(omega0 * .) -- the /omega0 exists purely
        to cancel the omega0 inside the sine. The FINAL layer of each axis
        network is linear and feeds the bilinear core product, NOT a sine, so
        that division does not apply to it. Including it (as an earlier version
        of this class did) shrinks both factor matrices by omega0 = 30, which
        leaves the pre-activation field with a standard deviation of ~5e-5 at
        initialization -- effectively constant -- against a target that needs a
        swing of ~10 in logit space. The model then never moves at all.
        """
        with torch.no_grad():
            first_fan_in = layers[0].weight.shape[1]
            layers[0].weight.uniform_(-1 / first_fan_in, 1 / first_fan_in)

            # sine-activated hidden layers: standard SIREN bound
            for layer in layers[1:-1]:
                fan_in = layer.weight.shape[1]
                bound = np.sqrt(6 / fan_in) / self.omega0
                layer.weight.uniform_(-bound, bound)

            # linear feature layer feeding the bilinear form: no omega0 division
            fan_in = layers[-1].weight.shape[1]
            bound = np.sqrt(6 / fan_in)
            layers[-1].weight.uniform_(-bound, bound)

    def init_siren_weights(self, core_init_std=1e-3):
        
        self.init_siren_stack(self.x_layers)
        self.init_siren_stack(self.y_layers)

        with torch.no_grad():
            # initialize core with small values so no random voidness occurs at the beginning
            self.core.normal_(mean=0.0, std=core_init_std)

            self.raw_bias.fill_(self.final_bias)

    def axis_forward(self, coord_1d, layers):
        """
        one-coordinate forward pass
        Input: coord_1d: [N, 1]
        Output: features: [N, rank]
        """
        x = coord_1d
        for layer in layers[:-1]:
            x = torch.sin(self.omega0 * layer(x))

        # Important!!!: final layer is linear because it produces feature coefficients, not voidness or gamma
        x = layers[-1](x)
        return x

    def gamma_from_raw(self, raw):
        if self.output_mode == "voidness":
            p_void = torch.sigmoid(raw)
            return 1.0 - (1.0 - self.gamma0) * p_void

        if self.output_mode == "direct_gamma":
            return self.gamma0 + (1.0 - self.gamma0) * torch.sigmoid(raw)
        raise ValueError(f"Unknown INR output_mode: {self.output_mode}")


    def forward(self, coords):
 
        x_coord = coords[:, 0:1]
        y_coord = coords[:, 1:2]

        fx = self.axis_forward(x_coord, self.x_layers)  # [N, rank]
        fy = self.axis_forward(y_coord, self.y_layers)  # [N, rank]

        # raw_i = fx_i^T C fy_i
        #
        # Equivalent readable form:
        #     tmp = fx @ self.core          # [N, rank]
        #     raw = (tmp * fy).sum(dim=1)   # [N]
        raw = ((fx @ self.core) * fy).sum(dim=1, keepdim=True)

        raw = raw + self.raw_bias

        return self.gamma_from_raw(raw)



class INR_MPE(torch.nn.Module):
    """
    Multi-grid Parametric Encoding INR for MPE-FWI.
    Concept:
        coords -> multi-resolution trainable grid features -> compact MLP
               -> raw voidness logits -> gamma
    """

    def __init__(
        self,
        gamma0,
        num_levels=16,
        base_resolution=50,
        per_level_scale=1.05,
        features_per_level=2,
        hidden_features=64,
        hidden_layers=2,
        output_mode="direct_gamma",
        final_bias=3.0,
        grid_init_std=1e-4,
        align_corners=True,
        swap_grid_coords=False,
        grid_aspect=1.0,
    ):
        super().__init__()

        self.gamma0 = gamma0
        self.num_levels = num_levels
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.features_per_level = features_per_level
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.output_mode = output_mode
        self.final_bias = final_bias
        self.grid_init_std = grid_init_std
        self.align_corners = align_corners
        self.swap_grid_coords = swap_grid_coords
        self.grid_aspect = float(grid_aspect)

        # Compute grid resolution for each level.
        # Paper settings: 16 levels, base resolution 50, per-level scale 1.05.
        self.resolutions = [
            int(round(base_resolution * (per_level_scale ** level)))
            for level in range(num_levels)
        ]

        # Trainable feature grids of shape
        #     [1, features_per_level, H, W]
        self.grids = torch.nn.ParameterList()
        for res in self.resolutions:
            height, width = self._grid_shape(res)
            grid = torch.nn.Parameter(
                grid_init_std * torch.randn(
                    1,
                    features_per_level,
                    height,
                    width,
                )
            )
            self.grids.append(grid)

        # Compact MLP after multi-resolution grid encoding
        encoded_dim = num_levels * features_per_level

        layers = []
        in_features = encoded_dim

        for _ in range(hidden_layers):
            layers.append(torch.nn.Linear(in_features, hidden_features))
            layers.append(torch.nn.ReLU())
            in_features = hidden_features

        layers.append(torch.nn.Linear(in_features, 1))

        self.mlp = torch.nn.Sequential(*layers)

        self.init_mlp_final_layer()

    def init_mlp_final_layer(self):
        """
        Initialize the final MLP layer so that initial gamma is close to 1.
        """
        final_layer = None
        for module in reversed(self.mlp):
            if isinstance(module, torch.nn.Linear):
                final_layer = module
                break

        if final_layer is None:
            raise RuntimeError("No final Linear layer found in INR_MPE.mlp")

        with torch.no_grad():
            # Small final weights help avoid random patterns
            final_layer.weight.normal_(0.0, 1e-3)
            final_layer.bias.fill_(self.final_bias)

    def _grid_shape(self, res):
        """
        Feature-grid (H, W) for a level of nominal resolution `res`.

        `F.grid_sample` maps coords[..., 0] to the W axis and coords[..., 1] to
        the H axis, so W tracks the first spatial coordinate and H the second.
        The simulation domain is not square (256 x 128), so allocating res x res
        grids -- as an earlier version did -- gave the two axes effective
        resolutions differing by the domain aspect ratio. `grid_aspect` is
        H/W; set it to (Ny+1)/(Nx+1) to keep the feature cells square. It
        defaults to 1.0, which reproduces the old square behaviour.
        """
        width = max(1, int(res))
        height = max(1, int(round(res * self.grid_aspect)))
        return height, width

    def sample_grid_features(self, coords):
        """
        Sample trainable features from all resolution levels.
        Input: coords: [N, 2], expected in [-1, 1]
        Output: encoded: [N, num_levels * features_per_level]
        """
        if coords.ndim != 2 or coords.shape[-1] != 2:
            raise ValueError(
                f"coords must have shape [N, 2], got {tuple(coords.shape)}"
            )

        sample_coords = coords
        if self.swap_grid_coords:
            sample_coords = sample_coords[:, [1, 0]]

        # grid_sample expects [B, H_out, W_out, 2].
        # We sample N points as H_out=N, W_out=1.
        sample_grid = sample_coords.view(1, -1, 1, 2)

        features = []

        for grid in self.grids:
            sampled = F.grid_sample(
                grid,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=self.align_corners,
            )

            # sampled shape: [1, features_per_level, N, 1]
            # Convert to: [N, features_per_level]
            sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)

            features.append(sampled)

        encoded = torch.cat(features, dim=-1)
        return encoded

    def gamma_from_raw(self, raw):
        if self.output_mode == "voidness":
            p_void = torch.sigmoid(raw)
            return 1.0 - (1.0 - self.gamma0) * p_void

        if self.output_mode == "direct_gamma":
            return self.gamma0 + (1.0 - self.gamma0) * torch.sigmoid(raw)

        raise ValueError(f"Unknown INR_MPE output_mode: {self.output_mode}")

    def forward(self, coords):
     
        encoded = self.sample_grid_features(coords)
        raw = self.mlp(encoded)
        return self.gamma_from_raw(raw)


class INR_MPE_CENTERED(torch.nn.Module):
    """
    MPE-INR with centered residual logits.
    Difference from INR_MPE: raw = final_bias + (MLP(encoded) - mean(MLP(encoded))).
    """

    def __init__(
        self,
        gamma0,
        num_levels=16,
        base_resolution=50,
        per_level_scale=1.05,
        features_per_level=2,
        hidden_features=64,
        hidden_layers=2,
        output_mode="direct_gamma",
        final_bias=3.0,
        grid_init_std=1e-2,
        align_corners=True,
        swap_grid_coords=False,
        grid_aspect=1.0,
    ):
        super().__init__()

        self.gamma0 = gamma0
        self.num_levels = num_levels
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.features_per_level = features_per_level
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.output_mode = output_mode
        self.grid_init_std = grid_init_std
        self.align_corners = align_corners
        self.swap_grid_coords = swap_grid_coords
        self.grid_aspect = float(grid_aspect)

        # Non-trainable final bias that sets the intact-material base logit.
        self.register_buffer("final_bias", torch.tensor(float(final_bias)))

        # Compute grid resolution for each level.
        # Paper settings: 16 levels, base resolution 50, per-level scale 1.05.
        self.resolutions = [
            int(round(base_resolution * (per_level_scale ** level)))
            for level in range(num_levels)
        ]

        # Trainable feature grids of shape
        #     [1, features_per_level, H, W]
        self.grids = torch.nn.ParameterList()
        for res in self.resolutions:
            height, width = self._grid_shape(res)
            grid = torch.nn.Parameter(
                grid_init_std * torch.randn(
                    1,
                    features_per_level,
                    height,
                    width,
                )
            )
            self.grids.append(grid)

        # Compact MLP after multi-resolution grid encoding
        encoded_dim = num_levels * features_per_level

        layers = []
        in_features = encoded_dim

        for _ in range(hidden_layers):
            layers.append(torch.nn.Linear(in_features, hidden_features))
            layers.append(torch.nn.ReLU())
            in_features = hidden_features

        layers.append(torch.nn.Linear(in_features, 1))

        self.mlp = torch.nn.Sequential(*layers)

        self.init_mlp_final_layer()

    def init_mlp_final_layer(self):
        """Initialize the final MLP layer for residual-logit prediction."""
        final_layer = None
        for module in reversed(self.mlp):
            if isinstance(module, torch.nn.Linear):
                final_layer = module
                break

        if final_layer is None:
            raise RuntimeError("No final Linear layer found in INR_MPE_Centered.mlp")

        with torch.no_grad():
            # final_bias supplies the base level; this layer predicts residuals.
            final_layer.weight.normal_(0.0, 1e-3)
            final_layer.bias.zero_()

    def _grid_shape(self, res):
        """
        Feature-grid (H, W) for a level of nominal resolution `res`.

        `F.grid_sample` maps coords[..., 0] to the W axis and coords[..., 1] to
        the H axis, so W tracks the first spatial coordinate and H the second.
        The simulation domain is not square (256 x 128), so allocating res x res
        grids -- as an earlier version did -- gave the two axes effective
        resolutions differing by the domain aspect ratio. `grid_aspect` is
        H/W; set it to (Ny+1)/(Nx+1) to keep the feature cells square. It
        defaults to 1.0, which reproduces the old square behaviour.
        """
        width = max(1, int(res))
        height = max(1, int(round(res * self.grid_aspect)))
        return height, width

    def sample_grid_features(self, coords):
        """
        Sample trainable features from all resolution levels.
        Input: coords: [N, 2], expected in [-1, 1]
        Output: encoded: [N, num_levels * features_per_level]
        """
        if coords.ndim != 2 or coords.shape[-1] != 2:
            raise ValueError(
                f"coords must have shape [N, 2], got {tuple(coords.shape)}"
            )

        sample_coords = coords
        if self.swap_grid_coords:
            sample_coords = sample_coords[:, [1, 0]]

        # grid_sample expects [B, H_out, W_out, 2].
        # We sample N points as H_out=N, W_out=1.
        sample_grid = sample_coords.view(1, -1, 1, 2)

        features = []

        for grid in self.grids:
            sampled = F.grid_sample(
                grid,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=self.align_corners,
            )

            # sampled shape: [1, features_per_level, N, 1]
            # Convert to: [N, features_per_level]
            sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)

            features.append(sampled)

        encoded = torch.cat(features, dim=-1)
        return encoded

    def gamma_from_raw(self, raw):
        if self.output_mode == "voidness":
            p_void = torch.sigmoid(raw)
            return 1.0 - (1.0 - self.gamma0) * p_void

        if self.output_mode == "direct_gamma":
            return self.gamma0 + (1.0 - self.gamma0) * torch.sigmoid(raw)

        raise ValueError(f"Unknown INR_MPE_Centered output_mode: {self.output_mode}")

    def forward(self, coords):
        encoded = self.sample_grid_features(coords)
        residual = self.mlp(encoded)
        residual = residual - residual.mean(dim=0, keepdim=True)
        raw = self.final_bias + residual
        return self.gamma_from_raw(raw)


class INR_IG(torch.nn.Module):
    """
    Integrated Grid-INR material-field parametrization for IG-FWI.
    Concept:
    1. Multi-resolution trainable feature grids, as in MPE-FWI.
    2. A sinusoidal coordinate feature network, as in SIREN/INR-FWI.
    3. A compact fusion MLP that maps concatenated features to raw voidness logits.
    
    Paper-style feature fusion:
        v(x) = sqrt(alpha) * h(x) concat sqrt(1 - alpha) * I(x)

    where:
        h(x) = multi-resolution grid features
        I(x) = sinusoidal INR features
    """

    def __init__(
        self,
        gamma0,
        # Fusion balance
        alpha=0.5,
        # MPE / grid branch
        num_levels=16,
        base_resolution=50,
        per_level_scale=1.05,
        features_per_level=2,
        grid_init_std=1e-4,
        align_corners=True,
        swap_grid_coords=False,
        grid_aspect=1.0,
        # SIREN feature branch
        siren_hidden_features=128,
        siren_hidden_layers=2,
        siren_out_features=128,
        omega0=30,
        # Fusion MLP
        fusion_hidden_features=64,
        fusion_hidden_layers=2,
        # Output
        output_mode="direct_gamma",
        final_bias=3.0,
        # Std of the fusion MLP's final-layer weights. This is the single lever
        # controlling how much gradient reaches the grid and SIREN branches:
        # d(raw)/d(branch feature) is linear in it. The original 1e-3 keeps the
        # initial output pinned to final_bias, but starves both branches --
        # measured grid-branch gradients fell to 3.4e-10, below Adam's eps.
        fusion_init_std=1e-3,
        # Rescale the fused feature vector to unit RMS before the fusion MLP.
        # Without it the MLP receives features of RMS ~0.06 and, after three
        # layers of default-scaled weights, emits `raw` with a standard deviation
        # of ~4e-3 -- so gamma varies by ~1e-4 against a target that varies by
        # ~1e-1. The model then starts as a near-constant field and cannot move.
        # This is the same failure mode as INR_LR's mis-transplanted init.
        feature_norm=False,
    ):
        super().__init__()

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.gamma0 = gamma0
        self.alpha = alpha

        self.num_levels = num_levels
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.features_per_level = features_per_level
        self.grid_init_std = grid_init_std
        self.align_corners = align_corners
        self.swap_grid_coords = swap_grid_coords
        self.grid_aspect = float(grid_aspect)

        self.siren_hidden_features = siren_hidden_features
        self.siren_hidden_layers = siren_hidden_layers
        self.siren_out_features = siren_out_features
        self.omega0 = omega0

        self.fusion_hidden_features = fusion_hidden_features
        self.fusion_hidden_layers = fusion_hidden_layers

        self.output_mode = output_mode
        self.final_bias = final_bias
        self.fusion_init_std = float(fusion_init_std)
        self.feature_norm = bool(feature_norm)

        # ------------------------------------------------------------
        # MPE branch
        # ------------------------------------------------------------
        self.resolutions = [
            int(round(base_resolution * (per_level_scale ** level)))
            for level in range(num_levels)
        ]

        self.grids = torch.nn.ParameterList()
        for res in self.resolutions:
            height, width = self._grid_shape(res)
            grid = torch.nn.Parameter(
                grid_init_std * torch.randn(
                    1,
                    features_per_level,
                    height,
                    width,
                )
            )
            self.grids.append(grid)

        self.grid_feature_dim = num_levels * features_per_level

        # ------------------------------------------------------------
        # SIREN feature branch
        # ------------------------------------------------------------
        self.siren_layers = torch.nn.ModuleList()

        in_features = 2
        for _ in range(siren_hidden_layers):
            self.siren_layers.append(
                torch.nn.Linear(in_features, siren_hidden_features)
            )
            in_features = siren_hidden_features

        # Final SIREN branch layer produces features, not gamma
        self.siren_layers.append(
            torch.nn.Linear(in_features, siren_out_features)
        )

        # ------------------------------------------------------------
        # Fusion MLP
        # ------------------------------------------------------------
        fusion_input_dim = self.grid_feature_dim + siren_out_features

        fusion_layers = []
        in_features = fusion_input_dim

        for _ in range(fusion_hidden_layers):
            fusion_layers.append(
                torch.nn.Linear(in_features, fusion_hidden_features)
            )
            fusion_layers.append(torch.nn.ReLU())
            in_features = fusion_hidden_features

        fusion_layers.append(torch.nn.Linear(in_features, 1))

        self.fusion_mlp = torch.nn.Sequential(*fusion_layers)

        self.init_weights()

    def init_siren_stack(self):
    
        with torch.no_grad():
            first_fan_in = self.siren_layers[0].weight.shape[1]
            self.siren_layers[0].weight.uniform_(
                -1 / first_fan_in,
                1 / first_fan_in,
            )

            for layer in self.siren_layers[1:]:
                fan_in = layer.weight.shape[1]
                bound = np.sqrt(6 / fan_in) / self.omega0
                layer.weight.uniform_(-bound, bound)

    def init_fusion_final_layer(self):
        """
        Initialize final fusion layer so that initial gamma is close to 1.
        """
        final_layer = None

        for module in reversed(self.fusion_mlp):
            if isinstance(module, torch.nn.Linear):
                final_layer = module
                break

        if final_layer is None:
            raise RuntimeError("No final Linear layer found in INR_IG.fusion_mlp")

        with torch.no_grad():
            # Make the initial output mostly controlled by the bias.
            # This prevents random grid/SIREN features from producing
            # strong random void patterns at epoch 0, while still allowing
            # immediate gradients to reach both feature branches.
            final_layer.weight.normal_(0.0, self.fusion_init_std)
            final_layer.bias.fill_(self.final_bias)

    def init_weights(self):
        self.init_siren_stack()
        self.init_fusion_final_layer()

    def _grid_shape(self, res):
        """
        Feature-grid (H, W) for a level of nominal resolution `res`.

        `F.grid_sample` maps coords[..., 0] to the W axis and coords[..., 1] to
        the H axis, so W tracks the first spatial coordinate and H the second.
        The simulation domain is not square (256 x 128), so allocating res x res
        grids -- as an earlier version did -- gave the two axes effective
        resolutions differing by the domain aspect ratio. `grid_aspect` is
        H/W; set it to (Ny+1)/(Nx+1) to keep the feature cells square. It
        defaults to 1.0, which reproduces the old square behaviour.
        """
        width = max(1, int(res))
        height = max(1, int(round(res * self.grid_aspect)))
        return height, width

    def sample_grid_features(self, coords):
        if coords.ndim != 2 or coords.shape[-1] != 2:
            raise ValueError(
                f"coords must have shape [N, 2], got {tuple(coords.shape)}"
            )

        sample_coords = coords

        if self.swap_grid_coords:
            sample_coords = sample_coords[:, [1, 0]]

        # grid_sample expects [B, H_out, W_out, 2].
        # We sample N points as H_out=N, W_out=1.
        sample_grid = sample_coords.view(1, -1, 1, 2)

        features = []

        for grid in self.grids:
            sampled = F.grid_sample(
                grid,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=self.align_corners,
            )

            # sampled shape: [1, features_per_level, N, 1]
            #
            # Convert to: [N, features_per_level]
            sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)

            features.append(sampled)

        return torch.cat(features, dim=-1)

    def normalize_features(self, features):
        """Scale the fused feature vector to unit RMS (no learnable parameters)."""
        if not self.feature_norm:
            return features
        rms = features.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return features / torch.clamp(rms, min=1e-8)

    def siren_features(self, coords):
        x = coords

        for layer in self.siren_layers[:-1]:
            x = torch.sin(self.omega0 * layer(x))

        # Final SIREN branch layer is linear because it outputs features.
        x = self.siren_layers[-1](x)

        return x

    def gamma_from_raw(self, raw):
        if self.output_mode == "voidness":
            p_void = torch.sigmoid(raw)
            return 1.0 - (1.0 - self.gamma0) * p_void

        if self.output_mode == "direct_gamma":
            return self.gamma0 + (1.0 - self.gamma0) * torch.sigmoid(raw)

        raise ValueError(f"Unknown INR_IG output_mode: {self.output_mode}")

    def forward(self, coords):
        grid_feat = self.sample_grid_features(coords)
        siren_feat = self.siren_features(coords)

        # Paper-style feature weighting:
        #  v = sqrt(alpha) h(x) concat sqrt(1-alpha) I(x)
        a = float(self.alpha)

        combined = torch.cat(
            [
                np.sqrt(a) * grid_feat,
                np.sqrt(1.0 - a) * siren_feat,
            ],
            dim=-1,
        )
        combined = self.normalize_features(combined)

        raw = self.fusion_mlp(combined)

        return self.gamma_from_raw(raw)


class INR_IG_CENTERED(INR_IG):
    """
    IG-INR with centered residual logits.
    Difference from INR_IG: raw = final_bias + (fusion_mlp(features) - mean(fusion_mlp(features))).
    """

    def __init__(self, *args, **kwargs):
        final_bias = kwargs.get("final_bias", 3.0)
        super().__init__(*args, **kwargs)
        del self.final_bias
        self.register_buffer("final_bias", torch.tensor(float(final_bias)))

    def init_fusion_final_layer(self):
        """Initialize final fusion layer for residual-logit prediction."""
        final_layer = None

        for module in reversed(self.fusion_mlp):
            if isinstance(module, torch.nn.Linear):
                final_layer = module
                break

        if final_layer is None:
            raise RuntimeError("No final Linear layer found in INR_IG_Centered.fusion_mlp")

        with torch.no_grad():
            # final_bias supplies the base level; this layer predicts residuals.
            final_layer.weight.normal_(0.0, self.fusion_init_std)
            final_layer.bias.zero_()

    def forward(self, coords):
        grid_feat = self.sample_grid_features(coords)
        siren_feat = self.siren_features(coords)
        a = float(self.alpha)

        combined = torch.cat(
            [
                np.sqrt(a) * grid_feat,
                np.sqrt(1.0 - a) * siren_feat,
            ],
            dim=-1,
        )

        combined = self.normalize_features(combined)
        residual = self.fusion_mlp(combined)
        residual = residual - residual.mean(dim=0, keepdim=True)
        raw = self.final_bias + residual
        return self.gamma_from_raw(raw)
    
    
    
###################################################################################################################################################################

def clipInputData(inputData, clipBC):
    if clipBC == 0:
        return inputData
    else:
        return torch.nn.Upsample(size=(inputData.shape[2], inputData.shape[3]))(
            inputData[:, :, clipBC:-clipBC, clipBC:-clipBC]
        )


class ConstantAnsatz(torch.nn.Module):
    def __init__(self, Nx, Ny):
        super().__init__()

        self.coeff = torch.nn.Parameter(
            torch.ones((1, 1, Nx + 1, Ny + 1), requires_grad=True)
        )

    def forward(self, x):
        return self.coeff

