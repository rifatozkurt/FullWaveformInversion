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
        for i in range(len(self.channels) - 1):
            x = self.upsample(x)
            x = self.convolutionsUp[self.numberOfConvolutionsPerBlock * i](
                torch.cat((x, x_[-(i + 1)]), 1)
            )
            if self.bnorm == True:
                x = self.bnormsUp[self.numberOfConvolutionsPerBlock * i](x)
            x = self.activationsUp[self.numberOfConvolutionsPerBlock * i](x)
            for j in range(1, self.numberOfConvolutionsPerBlock):
                x = self.convolutionsUp[self.numberOfConvolutionsPerBlock * i + j](x)
                x = self.activationsUp[self.numberOfConvolutionsPerBlock * i + j](x)

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
        output_mode="voidness",
        final_bias=-5.0,
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
        if self.output_mode == "voidness":
            # The INR predicts voidness logits, not gamma directly.
            # p_void = sigmoid(raw) is the local probability/strength of damage.
            # gamma = 1 - (1 - gamma_min) * p_void keeps intact material as the
            # default state when final_bias is negative.
            p_void = torch.sigmoid(raw)
            return 1.0 - (1.0 - self.gamma0) * p_void
        if self.output_mode == "direct_gamma":
            return self.gamma0 + (1.0 - self.gamma0) * torch.sigmoid(raw)
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
        output_mode="voidness",
        final_bias=-5.0,
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
    
    
class INR_LR(torch.nn.Module):
    """
    Low-rank INR material-field parametrization for LR-FWI.
    this model uses two axis-wise SIREN networks:
        x -> F_x(x) in R^rank
        y -> F_y(y) in R^rank
    and a trainable core matrix C:
        raw(x, y) = F_x(x)^T C F_y(y) + final_bias
    """

    def __init__(
        self,
        rank,
        hidden_features,
        hidden_layers,
        gamma0,
        omega0=30,
        output_mode="voidness",
        final_bias=-5.0,
        core_init_std=1e-3,
    ):
        super().__init__()

        self.rank = rank
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
        self.x_layers.append(torch.nn.Linear(in_features, rank))

        # y network: input dimension 1
        in_features = 1
        for _ in range(hidden_layers):
            self.y_layers.append(torch.nn.Linear(in_features, hidden_features))
            in_features = hidden_features
        self.y_layers.append(torch.nn.Linear(in_features, rank))

        # Trainable low-rank core matrix C.
        self.core = torch.nn.Parameter(torch.empty(rank, rank))  # [rank, rank]

        # Trainable scalar bias added to raw voidness logits.
        # !!! Initialized negative so that gamma initializes around 1
        self.raw_bias = torch.nn.Parameter(torch.tensor(float(final_bias)))

        self.init_siren_weights(core_init_std=core_init_std)

    def init_siren_stack(self, layers):
    
        with torch.no_grad():
            first_fan_in = layers[0].weight.shape[1]
            layers[0].weight.uniform_(-1 / first_fan_in, 1 / first_fan_in)

            for layer in layers[1:]:
                fan_in = layer.weight.shape[1]
                bound = np.sqrt(6 / fan_in) / self.omega0
                layer.weight.uniform_(-bound, bound)

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
        output_mode="voidness",
        final_bias=-5.0,
        grid_init_std=1e-4,
        align_corners=True,
        swap_grid_coords=False,
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
            grid = torch.nn.Parameter(
                grid_init_std * torch.randn(
                    1,
                    features_per_level,
                    res,
                    res,
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
        # SIREN feature branch
        siren_hidden_features=128,
        siren_hidden_layers=2,
        siren_out_features=128,
        omega0=30,
        # Fusion MLP
        fusion_hidden_features=64,
        fusion_hidden_layers=2,
        # Output
        output_mode="voidness",
        final_bias=-5.0,
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

        self.siren_hidden_features = siren_hidden_features
        self.siren_hidden_layers = siren_hidden_layers
        self.siren_out_features = siren_out_features
        self.omega0 = omega0

        self.fusion_hidden_features = fusion_hidden_features
        self.fusion_hidden_layers = fusion_hidden_layers

        self.output_mode = output_mode
        self.final_bias = final_bias

        # ------------------------------------------------------------
        # MPE branch
        # ------------------------------------------------------------
        self.resolutions = [
            int(round(base_resolution * (per_level_scale ** level)))
            for level in range(num_levels)
        ]

        self.grids = torch.nn.ParameterList()
        for res in self.resolutions:
            grid = torch.nn.Parameter(
                grid_init_std * torch.randn(
                    1,
                    features_per_level,
                    res,
                    res,
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
            final_layer.weight.normal_(0.0, 1e-3)
            final_layer.bias.fill_(self.final_bias)

    def init_weights(self):
        self.init_siren_stack()
        self.init_fusion_final_layer()

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

        raw = self.fusion_mlp(combined)

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
