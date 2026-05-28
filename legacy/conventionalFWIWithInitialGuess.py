# Import necessary libraries
import os

# Change the working directory to the script's directory
path_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_dir)

import torch
import time
import FiniteDifferencePyTorchConv as FiniteDifference
import AdjointMethod
import Utilities
import pandas as pd
import numpy as np
import NeuralNetwork as NN
import matplotlib.pyplot as plt
import matplotlib as mpl

# Matplotlib settings
mpl.rcParams.update({'font.size': 14})
mpl.rc('image', cmap='coolwarm')

# Source term parameters
frequency = 500000
cycles = 2
amplitude = 1e12

# Set manual seed for reproducibility and determine the device
torch.manual_seed(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

# Read settings and hyperparameters from CSV files
settings = pd.read_csv("settings2D.csv")

# Extract simulation parameters from settings
Lx = settings.Lx[0]
Ly = settings.Ly[0]
Nx = settings.Nx[0]
Ny = settings.Ny[0]
dx = Lx / Nx
dy = Ly / Ny
dt = settings.dt[0]
N = settings.N[0]
gamma0 = settings.gamma0[0]
rho = settings.rho[0]
c = settings.c[0]
numberOfSources = settings.numberOfSources[0]
distanceBetweenSources = settings.distanceBetweenSources[0]
distanceBetweenSensors = settings.distanceBetweenSensors[0]

# Get source and sensor locations
sourceLocationsx, sourceLocationsy = Utilities.getSourceLocations(Nx, Ny, distanceBetweenSources, numberOfSources)
selx, sely = Utilities.getSensorLocations(Nx, Ny, distanceBetweenSensors, sourceLocationsx)
numberOfSensors = len(selx)
print("Number of sensors: {:d}".format(len(selx)))

# Prepare initial condition terms for the forward solver
u0 = torch.zeros((numberOfSources, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device)
u1 = torch.zeros((numberOfSources, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device)

# Conventional FWI with initial guess from pretrained NN 

# Parameters of pretrained model
destinationFolder = "data_casestudy"
modelType = 'Unet'
torch.manual_seed(99)
pretrain_samples = 800
epochs_pretrain = 100
cases = np.linspace(1, 4, 4, dtype=int)

for case in cases:
    print(case)
    gamma = torch.ones((1, 1, Nx + 3, Ny + 3), dtype=torch.float32)
    initialGradient = torch.ones((1, 1, Nx + 1, Ny + 1), dtype=torch.float32)
    um = torch.zeros((1, numberOfSources, numberOfSensors, N),
                     dtype=torch.float32)

    gamma[0, 0, 1:-1, 1:-1] = torch.tensor(
        pd.read_hdf(destinationFolder + "/material" + str(case) + ".h5").values).to(device)

    initialGradient[0, 0, :, :] = torch.tensor(
        pd.read_hdf(destinationFolder + "/gradient" + str(case) + ".h5").values).to(device)

    um[0, :, :, :] = torch.tensor(
        pd.read_hdf(destinationFolder + "/measurement" + str(case) + ".h5").values).view(1,
                                                                                         numberOfSources,
                                                                                         numberOfSensors, N).to(device)

    F = (torch.tensor(pd.read_hdf(destinationFolder + "/source.h5").values)
         .reshape((numberOfSources, 1, Nx + 3, Ny + 3, N))).to(device)

    NNchannels = [1, 16, 32, 64, 128]
    numberOfConvolutionsPerBlock = 2
    model = NN.Unet(NNchannels, numberOfConvolutionsPerBlock, gamma0, bnorm=True)
    model.load_state_dict(torch.load(
        "model_" + modelType + "_" + str(epochs_pretrain) + "_supervised_" + str(pretrain_samples) + "_channel_" + str(
            len(NNchannels)), map_location=device))

    forwardSolver = FiniteDifference.FiniteDifference(dt, dx, dy, c, rho, device=device)
    inputData = initialGradient
    inputData = (inputData - torch.amin(inputData, (2, 3), keepdim=True)) / (
            torch.amax(inputData, (2, 3), keepdim=True)
            - torch.amin(inputData, (2, 3), keepdim=True)) * 2 - 1
    inputData = inputData.to(device)

    # training
    lr = 5e-2
    alpha = 0.
    beta = 0.
    weightLrFactor = 1
    clipGrad = 1e-5
    epochs = 35
    costScaling = 1e12

    # u0 and u1 are required as input for the adjoint method
    u0 = torch.zeros((numberOfSources, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device)
    u1 = torch.zeros((numberOfSources, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device)

    # Defining the gamma as torch parameters to be optimized
    gammaPred = torch.nn.Parameter(torch.ones((1, 1, Nx + 3, Ny + 3), device=device))
    gammaPred.data[0, 0, 1:-1, 1:-1] = model(inputData)
    gammaPred.grad = torch.zeros_like(gammaPred, device=device)

    # defining optimizer
    optimizer = torch.optim.Adam((gammaPred,), lr=lr)
    lr_lambda = lambda epoch: (beta * epoch + 1) ** alpha
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # initializing the forward solver
    forwardSolver = FiniteDifference.FiniteDifference(dt, dx, dy, c, rho, device=device)

    # Variables in which the cost, MSE and gamma over iterations will be stored over iterations
    historyCost = np.zeros(epochs)
    historyMSE = np.zeros(epochs)
    historyGamma = np.zeros((epochs + 1, Nx + 3, Ny + 3))

    start0 = time.perf_counter()
    start = start0
    historyGamma[0] = gammaPred.detach().cpu()
    for epoch in range(epochs):
        cost, gradient = AdjointMethod.getAdjointGradient(forwardSolver, u0, u1, c, rho, gammaPred.detach(), F, Nx, dx,
                                                          Ny, dy, N, dt,
                                                          numberOfSources, um.to(device), selx, sely, device)

        gammaPred.grad[0, 0, 1:-1, 1:-1] = gradient * costScaling

        # setting the gradient at the senors and source locations as zero since it causes problem of exploding cost function sometimes
        gammaPred.grad[0, 0, selx, sely] = 0
        gammaPred.grad[0, 0, sourceLocationsx, sourceLocationsy] = 0
        gammaPred.backward(gammaPred.grad)

        torch.nn.utils.clip_grad_norm_(gammaPred, clipGrad)
        optimizer.step()
        scheduler.step()
        gammaPred.data = gammaPred.data.clamp(gamma0, 1)

        historyCost[epoch] = cost
        historyMSE[epoch] = 0.5 * torch.mean((gamma - gammaPred.cpu()) ** 2).detach().cpu()
        historyGamma[epoch + 1] = gammaPred.detach().cpu()

        elapsed_time = time.perf_counter() - start
        if epoch % 2 == 0:
            string = "Epoch: {}/{}\tCost function: {:.3E}\tMSE: {:.3E}\tElapsed time: {:2f}"
            print(string.format
                  (epoch, epochs - 1, historyCost[epoch], historyMSE[epoch], elapsed_time))
        start = time.perf_counter()

    # Plotting and saving figures
    fig, ax = plt.subplots(9, 1, figsize=(5, 27))
    for i in range(len(ax) - 2):
        ax[i].imshow(np.transpose(historyGamma[i * epochs // (len(ax) - 2), 1:-1, 1:-1]), vmin=0, vmax=1)
        ax[i].axis('off')

    ax[-2].imshow(np.transpose(historyGamma[-1, 1:-1, 1:-1]), vmin=0, vmax=1)
    ax[-2].axis('off')
    ax[-1].imshow(np.transpose(gamma[0, 0, 1:-1, 1:-1].detach().cpu()), vmin=0, vmax=1)
    ax[-1].axis('off')

    plt.savefig(destinationFolder + '/Figure/pretrained_conventionalFWI_' + str(pretrain_samples) + '_adjoint_' + str(
        case) + '.svg')
    # plt.show()
    plt.close()

    # Saving the gamma history over iterations
    pd.DataFrame(historyGamma.reshape(epochs + 1, -1)).to_hdf(
        destinationFolder + "/GammaHistory/pretrained_conventionalFWI_" + str(pretrain_samples) + "_gammaHistory" + str(
            case) + ".h5",
        key="f",
        index=False,
        mode="w",
        complevel=1)

    # Saving the cost history over iterations
    np.savetxt(
        destinationFolder + "/CostHistory/pretrained_conventionalFWI_" + str(pretrain_samples) + "_costHistory" + str(
            case) + ".txt",
        historyCost,
        delimiter=", ")

    # Saving the MSE history over iterations
    np.savetxt(
        destinationFolder + "/MSEHistory/pretrained_conventionalFWI_" + str(pretrain_samples) + "_mseHistory" + str(
            case) + ".txt",
        historyMSE,
        delimiter=", ")
