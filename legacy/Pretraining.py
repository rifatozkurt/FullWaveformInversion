import os

# Change the working directory to the script's directory
path_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_dir)

import torch
import time
import Utilities
import random
import pandas as pd
import numpy as np
import NeuralNetwork as NN
import FiniteDifferencePyTorchConv as FiniteDifference
from torch.utils.data import Dataset, DataLoader

# Define device for computation (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load settings and hyperparameters from CSV files
settings = pd.read_csv("settings2D.csv")

# Extract specific case settings
Lx = settings.Lx[0]  # domain dimensions
Ly = settings.Ly[0]
Nx = settings.Nx[0]
Ny = settings.Ny[0]
dx = Lx / Nx
dy = Ly / Ny
dt = settings.dt[0]  # time step size
N = settings.N[0]  # number of time steps

# Extract physical parameters
gamma0 = settings.gamma0[0]
rho = settings.rho[0]
c = settings.c[0]

# Extract source and sensor parameters
numberOfSources = settings.numberOfSources[0]
distanceBetweenSources = settings.distanceBetweenSources[0]
distanceBetweenSensors = settings.distanceBetweenSensors[0]

# Get source and sensor locations
sourceLocationsx, sourceLocationsy = Utilities.getSourceLocations(Nx, Ny, distanceBetweenSources, numberOfSources)
selx, sely = Utilities.getSensorLocations(Nx, Ny, distanceBetweenSensors, sourceLocationsx)
numberOfSensors = len(selx)

# Load and preprocess data for training
destinationFolder = "data"
numberOfSamples = 800
initialGradient = torch.zeros((numberOfSamples, 1, Nx + 1, Ny + 1), dtype=torch.float32, device=device)
gamma = torch.ones((numberOfSamples, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device)
um = torch.zeros((numberOfSamples, numberOfSources, numberOfSensors, N), dtype=torch.float32, device=device)
idx_numberOfSamples = random.sample(range(800), numberOfSamples)  # assuming there is a total of 1000 training data set

# Load initial gradient, material, and measurement data
for idx, file_idx in enumerate(idx_numberOfSamples):
    initialGradient[idx, 0] = torch.tensor(
        pd.read_hdf(destinationFolder + "/gradient" + str(file_idx) + ".h5").values).to(device).to(torch.float32)

    gamma[idx, 0, 1:-1, 1:-1] = torch.tensor(
        pd.read_hdf(destinationFolder + "/material" + str(file_idx) + ".h5").values).to(device).to(torch.float32)

    um = torch.tensor(
        pd.read_hdf(destinationFolder + "/measurement" + str(file_idx) + ".h5").values).to(device).to(torch.float32)

# hyperparameters
trainingType = "supervised"

# Neural network parameters
NNchannels = [1, 16, 32, 64, 128]
numberOfConvolutionsPerBlock = 2

batchSize = numberOfSamples // 10

# Normalize input data
inputData = initialGradient
inputData = (inputData - torch.amin(inputData, (2, 3), keepdim=True)) / (
        torch.amax(inputData, (2, 3), keepdim=True)
        - torch.amin(inputData, (2, 3), keepdim=True)) * 2 - 1

# Create dataset and dataloaders
dataset = NN.FWIDataset(inputData, gamma, device)
datasetTraining, datasetValidation = torch.utils.data.random_split(
    dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(2))

dataloaderTraining = DataLoader(datasetTraining, batch_size=batchSize)
dataloaderValidation = DataLoader(datasetValidation, batch_size=len(datasetValidation))

# Initialize model
torch.manual_seed(30)
model_type = "Unet"
torch.use_deterministic_algorithms(True)
model = NN.Unet(NNchannels, numberOfConvolutionsPerBlock, gamma0, bnorm=True)
NN.initWeights(model)
torch.nn.init.normal_(model.convolutionsUp[-1].weight, std=0.01, mean=0.7)
model.convolutionsUp[-1].bias.data.fill_(3)
model.to(device)

# Training parameters
lr = 5e-4
alpha = -0.5
beta = 0.2
epochs = 1
clipGrad = 5e-4
l2 = 1e-6
costScaling = 1e8

# Initialize optimizer and scheduler
optimizer = torch.optim.RMSprop(model.parameters(), lr)
weights = torch.ones(2, requires_grad=True, dtype=torch.float, device=device)
lr_lambda = lambda epoch: (beta * epoch + 1) ** alpha
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Initialize forward solver
forwardSolver = FiniteDifference.FiniteDifference(dt, dx, dy, c, rho, device=device)

# Initialize training history variables
trainingCostHistory = np.zeros(epochs)
trainingMSEHistory = np.zeros(epochs)
validationCostHistory = np.zeros(epochs)
start = time.perf_counter()
start0 = start
beta = 0

# Training loop
for epoch in range(epochs):
    model.train()
    for batch, sample in enumerate(dataloaderTraining):
        # batch.to(device)
        sample[0] = sample[0].to(device)
        sample[1] = sample[1].to(device)
        optimizer.zero_grad(set_to_none=True)

        # Training in a supervised manner where the output spatial
        # material distribution is the output and the error is a simple L2 loss

        gammaPred = torch.ones(
            (len(sample[0]), 1, Nx + 3, Ny + 3), device=device, dtype=torch.float32)
        gammaPred[:, :, 1:-1, 1:-1] = model(sample[0])
        cost = (0.5 * torch.mean((gammaPred - sample[1]) ** 2))
        cost.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clipGrad)
        optimizer.step()
        scheduler.step()
        trainingCostHistory[epoch] += cost.detach().cpu()

    trainingCostHistory[epoch] /= batch + 1
    trainingMSEHistory[epoch] /= batch + 1

    # Validation
    model.eval()
    sample = next(iter(dataloaderValidation))
    sample[0] = sample[0].to(device)
    sample[1] = sample[1].to(device)
    gammaPred = torch.ones(
        (len(sample[0]), 1, Nx + 3, Ny + 3), device=device, dtype=torch.float32)
    gammaPred[:, :, 1:-1, 1:-1] = model(sample[0])
    validationCostHistory[epoch] = 0.5 * torch.mean((gammaPred.detach().cpu() - sample[1].cpu()) ** 2)

    if epoch % 10 == 0:
        elapsed_time = time.perf_counter() - start
        string = "Epoch: {}/{}\tTraining Cost: {:.6E}\t Validation Cost: {:3E}\tElapsed time: {:2f} \t Input per sec: {:3f}"
        print(string.format(epoch, epochs - 1, trainingCostHistory[epoch], validationCostHistory[epoch], elapsed_time,
                            batchSize / elapsed_time))
        start = time.perf_counter()

# Save the pretrained model
torch.save(model.state_dict(), "model_" + model_type + "_" + str(epochs) + "_" + trainingType + "_" + str(
    numberOfSamples) + "_channel_" + str(len(NNchannels)))
