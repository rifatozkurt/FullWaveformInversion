# Import necessary libraries
import os

# Change the working directory to the script's directory
path_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_dir)

import torch
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import NeuralNetwork as NN
import AdjointMethod
import FiniteDifferencePyTorchConv as FiniteDifference
import Utilities
import matplotlib as mpl

# Matplotlib settings
mpl.rcParams.update({'font.size': 14})
mpl.rc('image', cmap='coolwarm')

# source term parameters
frequency = 500000
cycles = 2
amplitude = 1e12

# Set manual seed for reproducibility and determine the device
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

# Prepare initial conditions terms for the forward solver
u0 = torch.zeros((numberOfSources, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device)
u1 = torch.zeros((numberOfSources, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device)

cases = np.linspace(1, 4, 4, dtype=int)
destinationFolder = "data_casestudy"
forwardSolver = FiniteDifference.FiniteDifference(dt, dx, dy, c, rho, device=device)

# Constructing the folder structure
subfolders = ['Figure', 'GammaHistory', 'CostHistory', 'MSEHistory']

if destinationFolder not in os.listdir(os.getcwd()):
    os.mkdir(destinationFolder)

for sub_f in subfolders:
    if sub_f not in os.listdir(os.path.join(os.getcwd(), destinationFolder)):
        os.mkdir(os.path.join(destinationFolder, sub_f))

for case in cases:
    print(case)
    initialGradient = torch.ones((1, 1, Nx + 3, Ny + 3), dtype=torch.float32)
    um = torch.zeros((1, numberOfSources, numberOfSensors, N), dtype=torch.float32)
    gamma = torch.ones((1, 1, Nx + 3, Ny + 3), dtype=torch.float32)

    gamma[0, 0, 1:-1, 1:-1] = torch.tensor(
        pd.read_hdf(destinationFolder + "/material" + str(case) + ".h5").values).to(device)

    initialGradient[0, 0, 1:-1, 1:-1] = torch.tensor(
        pd.read_hdf(destinationFolder + "/gradient" + str(case) + ".h5").values).to(device)

    um[0, :, :, :] = torch.tensor(
        pd.read_hdf(destinationFolder + "/measurement" + str(case) + ".h5").values).view(1, numberOfSources,
                                                                                         numberOfSensors, N).to(device)

    F = (torch.tensor(pd.read_hdf(destinationFolder + "/source.h5").values)
         .reshape((numberOfSources, 1, Nx + 3, Ny + 3, N))).to(device)

    # Decoder network
    torch.manual_seed(50)
    model = NN.CNNGamma(gamma0)
    NN.initWeights(model)

    # Initialize the non-zero weights with specified mean and standard deviation
    torch.nn.init.normal_(model.convOut.weight, std=0.01)
    model.convOut.bias.data.fill_(3)
    model.to(device)

    # Generating random inputs. If no. of gird points are changed, the size of the input also needs to be changed accordingly.
    inputData = torch.randn((1, 128, 8, 4), device=device)
    torch.nn.init.trunc_normal_(inputData, mean=0.4, std=0.2, a=0.0, b=0.9)
    inputData = (inputData - torch.amin(inputData, (2, 3), keepdim=True)) / (
            torch.amax(inputData, (2, 3), keepdim=True)
            - torch.amin(inputData, (2, 3), keepdim=True)) * 2 - 1
    inputData.to(device)

    # Training hyperparameters
    lr = 5e-4
    alpha = -0.5
    beta = 0.2
    epochs = 35
    clipGrad = 5e-5
    l2 = 1e-6
    costScaling = 1e9

    # Prepare input data for the model
    gammaPred = torch.ones((1, 1, Nx + 3, Ny + 3), device=device, dtype=torch.float32)  # with ghost cells
    gammaPred[:, :, 2:-2, 2:-2] = model(inputData)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=l2)
    lr_lambda = lambda epoch: (beta * epoch + 1) ** alpha
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # training
    trainingCostHistory = np.zeros(epochs)
    validationCostHistory = np.zeros(epochs)

    costHistory = np.zeros(epochs)
    mseHistory = np.zeros(epochs)
    avgPrecision = np.zeros(epochs)
    gammaHistory = np.zeros((epochs + 1, Nx + 1, Ny + 1))
    snrHistory = np.zeros(epochs)
    start0 = time.perf_counter()
    start = start0
    gammaHistory[0] = gammaPred[0, 0, 1:-1, 1:-1].detach().cpu()

    for epoch in range(epochs):
        # Initialze and write the output of the NN
        gammaPred = torch.ones((1, 1, Nx + 3, Ny + 3), device=device, dtype=torch.float32)  # with ghost cells
        gammaPred[:, :, 2:-2, 2:-2] = model(inputData)
        gammaPred.grad = torch.zeros_like(gammaPred.detach(), device=device)

        # Adjoint gradient and the cost function
        cost, gradient = AdjointMethod.getAdjointGradient(
            forwardSolver, u0, u1, c, rho,
            gammaPred.detach(), F, Nx, dx, Ny, dy, N, dt,
            numberOfSources, um.to(device), selx, sely, device)
        gammaPred.grad[0, 0, 1:-1, 1:-1] = gradient * costScaling
        gammaPred.backward(gammaPred.grad)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipGrad)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        costHistory[epoch] = cost.detach().cpu()
        mseHistory[epoch] = 0.5 * torch.mean((gammaPred[0] - gamma.to(device)) ** 2).detach().cpu()
        gammaHistory[epoch + 1] = gammaPred[0, 0, 1:-1, 1:-1].detach().cpu()
        elapsed_time = time.perf_counter() - start

        if epoch % 2 == 0:
            string = "Epoch: {}/{}\t\tCost function: {:.3E}\t\tMSE: {:.3E}\t\tElapsed time: {:2f}"
            print(string.format(epoch, epochs - 1, costHistory[epoch], mseHistory[epoch], elapsed_time))
        start = time.perf_counter()

    # Plotting and saving figures over iterations
    fig, ax = plt.subplots(9, 1, figsize=(5, 27))
    for i in range(len(ax) - 2):
        ax[i].imshow(np.transpose(gammaHistory[i * epochs // (len(ax) - 2), 1:-1, 1:-1]), vmin=0, vmax=1)
        ax[i].axis("off")
    ax[-2].imshow(np.transpose(gammaHistory[-1, 1:-1, 1:-1]), vmin=0, vmax=1)
    ax[-2].axis("off")
    ax[-1].imshow(np.transpose(gamma[0, 0, 1:-1, 1:-1].detach().cpu()), vmin=0, vmax=1)
    ax[-1].axis("off")
    plt.savefig(destinationFolder + '/Figure/nonpretrained_' + 'samples_case' + str(case) + '.svg')
    plt.close()

    # Saving gamma over iterations
    pd.DataFrame(gammaHistory.reshape(epochs + 1, -1)).to_hdf(
        destinationFolder + "/GammaHistory/nonpretrained_gammaHistory_" + str(case) + ".h5",
        key="f",
        index=False,
        mode="w",
        complevel=1)

    # Saving cost over iterations
    np.savetxt(
        destinationFolder + "/CostHistory/nonpretrained_costHistory_" + str(case) + ".txt",
        costHistory,
        delimiter=", ")

    # Saving MSE over iterations
    np.savetxt(
        destinationFolder + "/MSEHistory/nonpretrained_mseHistory_" + str(case) + ".txt",
        mseHistory,
        delimiter=", ")
