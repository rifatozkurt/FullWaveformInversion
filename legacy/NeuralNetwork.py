import torch
from torch.utils.data import Dataset


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
