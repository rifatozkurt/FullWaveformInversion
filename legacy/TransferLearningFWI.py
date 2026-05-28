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
print("Number of sensors: {:d}".format(len(selx)))

# Prepare source term
F = Utilities.getSource(frequency, cycles, amplitude, sourceLocationsx, sourceLocationsy,
                        Nx, Ny, dx, dy, Lx, Ly, N, dt).to(device)
u0 = torch.zeros((numberOfSources, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device)
u1 = torch.zeros((numberOfSources, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device)

# Transfer learning FWI
pretrain_samples = [800]
cases = np.linspace(1, 4, 4, dtype=int)
numberOfSensors = len(selx)
epochs_pretrain = 100
destinationFolder = "data_casestudy"

# Constructing the folder structure
subfolders = ['Figure', 'GammaHistory', 'CostHistory', 'MSEHistory']

if destinationFolder not in os.listdir(os.getcwd()):
    os.mkdir(destinationFolder)

for sub_f in subfolders:
    if sub_f not in os.listdir(os.path.join(os.getcwd(), destinationFolder)):
        os.mkdir(os.path.join(destinationFolder, sub_f))

for sample in pretrain_samples:
    for case in cases:
        print(case)

        # Read the data
        initialGradient = torch.ones((1, 1, Nx + 1, Ny + 1), dtype=torch.float32)
        um = torch.zeros((1, numberOfSources, numberOfSensors, N), dtype=torch.float32)
        gamma = torch.ones((1, 1, Nx + 3, Ny + 3), dtype=torch.float32)

        gamma[0, 0, 1:-1, 1:-1] = torch.tensor(
            pd.read_hdf(destinationFolder + "/material" + str(case) + ".h5").values).to(device)

        initialGradient[0, 0] = torch.tensor(
            pd.read_hdf(destinationFolder + "/gradient" + str(case) + ".h5").values).to(device)

        um[0, :, :, :] = torch.tensor(
            pd.read_hdf(destinationFolder + "/measurement" + str(case) + ".h5").values).view(1, numberOfSources,
                                                                                             numberOfSensors, N).to(
            device)

        f = (torch.tensor(pd.read_hdf(destinationFolder + "/source.h5").values)
             .reshape((numberOfSources, 1, Nx + 3, Ny + 3, N))).to(device)

        # Load the model
        modelType = 'Unet'
        torch.manual_seed(99)

        # Set model parameters
        lr = 5e-4
        alpha = -0.5
        beta = 0.2
        epochs = 5
        clipGrad = 5e-5
        l2 = 1e-6
        costScaling = 1e10

        NNchannels = [1, 16, 32, 64, 128]
        numberOfConvolutionsPerBlock = 2
        model = NN.Unet(NNchannels, numberOfConvolutionsPerBlock, gamma0, bnorm=True)
        model.load_state_dict(torch.load(
            "model_" + modelType + "_" + str(epochs_pretrain) + "_supervised_" + str(sample) + "_channel_" + str(
                len(NNchannels)), map_location=device))
        model.to(device)

        # Prepare input data for the model
        gammaPred = torch.ones((1, 1, Nx + 3, Ny + 3), device=device, dtype=torch.float32)  # with ghost cells
        forwardSolver = FiniteDifference.FiniteDifference(dt, dx, dy, c, rho, device=device)
        inputData = initialGradient
        inputData = (inputData - torch.amin(inputData, (2, 3), keepdim=True)) / (
                torch.amax(inputData, (2, 3), keepdim=True)
                - torch.amin(inputData, (2, 3), keepdim=True)) * 2 - 1
        inputData = inputData.to(device)
        gammaPred[:, :, 1:-1, 1:-1] = model(inputData)  # initial guess

        # solver
        optimizer = torch.optim.Adam(model.parameters(), lr)
        lr_lambda = lambda epoch: (beta * epoch + 1) ** alpha
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # History variables
        trainingCostHistory = np.zeros(epochs)
        validationCostHistory = np.zeros(epochs)
        costHistory = np.zeros(epochs)
        mseHistory = np.zeros(epochs)
        gammaHistory = np.zeros((epochs + 1, Nx + 1, Ny + 1))
        gammaHistory[0] = gammaPred[0, 0, 1:-1, 1:-1].cpu().detach().numpy()

        start0 = time.perf_counter()
        start = start0
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            model.train()

            # Initialze and write the output of the NN
            gammaPred = torch.ones((1, 1, Nx + 3, Ny + 3), device=device, dtype=torch.float32)  # with ghost cells
            gammaPred[:, :, 1:-1, 1:-1] = model(inputData)
            gammaPred.grad = torch.zeros_like(gammaPred.detach(), device=device)

            # Adjoint gradient and the cost function
            cost, gradient = AdjointMethod.getAdjointGradient(forwardSolver, u0, u1, c, rho, gammaPred.detach(), F, Nx,
                                                              dx, Ny, dy, N, dt,
                                                              numberOfSources, um.to(device), selx, sely, device)
            gammaPred.grad[0, 0, 1:-1, 1:-1] = gradient * costScaling
            gammaPred.backward(gammaPred.grad)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipGrad)
            optimizer.step()
            scheduler.step()

            costHistory[epoch] = cost.detach().cpu()
            mseHistory[epoch] = 0.5 * torch.mean((gammaPred[0].cpu() - gamma) ** 2).detach().cpu()
            gammaHistory[epoch + 1] = gammaPred[0, 0, 1:-1, 1:-1].detach().cpu()

            elapsed_time = time.perf_counter() - start
            if epoch % 2 == 0:
                string = "Epoch: {}/{}\t\tCost function: {:.7E}\t\tMSE: {:.7E}\t\tElapsed time: {:2f}"
                print(string.format(epoch, epochs - 1, costHistory[epoch], mseHistory[epoch], elapsed_time))
            start = time.perf_counter()
        print("Total elapsed time: {:2f}".format(time.perf_counter() - start0))

        ##Plotting and saving figures over iterations
        fig, ax = plt.subplots(9, 1, figsize=(5, 27))
        for i in range(len(ax) - 2):
            ax[i].imshow(np.transpose(gammaHistory[i * epochs // (len(ax) - 2), 1:-1, 1:-1]), vmin=0, vmax=1)
            ax[i].axis('off')
        ax[-2].imshow(np.transpose(gammaHistory[-1, 1:-1, 1:-1]), vmin=0, vmax=1)
        ax[-2].axis('off')
        ax[-1].imshow(np.transpose(gamma[0, 0, 1:-1, 1:-1].detach().cpu()), vmin=0, vmax=1)
        ax[-1].axis("off")

        # plt.show()
        plt.savefig(destinationFolder + '/Figure/transferFWI_' + modelType + str(sample) + "_" + str(
            epochs_pretrain) + '_case' + str(case) + '.svg')
        plt.close()

        # Saving gamma over iterations
        pd.DataFrame(gammaHistory.reshape(epochs + 1, -1)).to_hdf(
            destinationFolder + "/GammaHistory/transferFWI_" + modelType + str(sample) + "_" + str(
                epochs_pretrain) + "_gamma" + str(case) + ".h5",
            key="f",
            index=False,
            mode="w",
            complevel=1)

        # Saving cost over iterations
        np.savetxt(
            destinationFolder + "/CostHistory/transferFWI_" + modelType + str(sample) + "_" + str(
                epochs_pretrain) + "_cost" + str(case) + ".txt",
            costHistory,
            delimiter=", ")

        # Saving MSE over iterations
        np.savetxt(
            destinationFolder + "/MSEHistory/transferFWI_" + modelType + str(sample) + "_" + str(
                epochs_pretrain) + "_mse" + str(case) + ".txt",
            mseHistory,
            delimiter=", ")
