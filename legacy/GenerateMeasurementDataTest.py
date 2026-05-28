# Change the directory
import os

path_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_dir)

import torch
import time
import FiniteDifferencePyTorchConv as FiniteDifference
import AdjointMethod
import Utilities
import pandas as pd
import numpy as np

# source term parameters
frequency = 500000
cycles = 2
amplitude = 1e12

torch.manual_seed(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read parameters from the settings file.
settings = pd.read_csv("settings2D.csv")

# case_grid for switching the parameters related to the forward solver.

Lx = settings.Lx[0]  # domain dimensions
Ly = settings.Ly[0]
Nx = settings.Nx[0]
Ny = settings.Ny[0]
dx = Lx / Nx
dy = Ly / Ny
dt = settings.dt[0]  # time step size
N = settings.N[0]  # number of time steps

gamma0 = settings.gamma0[0]
rho = settings.rho[0]
c = settings.c[0]

numberOfSources = settings.numberOfSources[0]
distanceBetweenSources = settings.distanceBetweenSources[0]
distanceBetweenSensors = settings.distanceBetweenSensors[0]
sourceLocationsx, sourceLocationsy = Utilities.getSourceLocations(Nx, Ny, distanceBetweenSources, numberOfSources)

selx, sely = Utilities.getSensorLocations(Nx, Ny, distanceBetweenSensors, sourceLocationsx)
print("Number of sensors: {:d}".format(len(selx)))

# second grid to avoid inverse crime
factorToAvoidInverseCrime = 2
Nx_ = Nx * factorToAvoidInverseCrime  # number of grid points (without ghost cells)
Ny_ = Ny * factorToAvoidInverseCrime  # number of grid points (without ghost cells)
dx_ = Lx / Nx_
dy_ = Ly / Ny_
dt_ = dt / factorToAvoidInverseCrime
N_ = N * factorToAvoidInverseCrime
distanceBetweenSources_ = distanceBetweenSources * factorToAvoidInverseCrime
distanceBetweenSensors_ = distanceBetweenSensors * factorToAvoidInverseCrime

u0_ = torch.zeros(
    (numberOfSources, 1, Nx_ + 3, Ny_ + 3), dtype=torch.float32, device=device)

u1_ = torch.zeros(
    (numberOfSources, 1, Nx_ + 3, Ny_ + 3), dtype=torch.float32, device=device)

forwardSolver_ = FiniteDifference.FiniteDifference(dt_, dx_, dy_, c, rho, device=device)

sourceLocationsx_, sourceLocationsy_ = Utilities.getSourceLocations(
    Nx_, Ny_, distanceBetweenSources_, numberOfSources)

selx_, sely_ = Utilities.getSensorLocations(
    Nx_, Ny_, distanceBetweenSensors_, sourceLocationsx_)

x = np.linspace(0 - dx, Lx + dx, Nx + 3)  # with ghost cells
x_ = np.linspace(0 - dx_, Lx + dx_, Nx_ + 3)  # with ghost cells
if np.sum((x[selx] - x_[selx_]) ** 2) > 1e-20:
    print("Error: sensor locations are unequal")

F_ = Utilities.getSource(frequency, cycles, amplitude,
                         sourceLocationsx_, sourceLocationsy_,
                         Nx_, Ny_, dx_, dy_, Lx, Ly, N_, dt_).to(device)

n_dmg = 1
# Set the destination folder where to write the files
destinationFolder = "dataTest"
u0 = torch.zeros((numberOfSources, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device)
u1 = torch.zeros((numberOfSources, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device)
forwardSolver = FiniteDifference.FiniteDifference(dt, dx, dy, c, rho, device=device)
F = Utilities.getSource(frequency, cycles, amplitude,
                        sourceLocationsx, sourceLocationsy,
                        Nx, Ny, dx, dy, Lx, Ly, N, dt).to(device)

if destinationFolder not in os.listdir(os.getcwd()):
    os.mkdir(destinationFolder)

numberOfCases = 2
for case in range(0, numberOfCases):
    print(case)
    gamma_ = torch.ones((1, 1, Nx_ + 3, Ny_ + 3)).to(device)

    for i in range(n_dmg):
        gamma_t, x0, y0, a, b, theta = Utilities.generateGamma(
            Nx_, Ny_, dx_, dy_, Lx, Ly, gamma0)
        gamma_ *= gamma_t.to(device)

    gamma_ = gamma_.to(device)

    # Generate measurements
    start = time.perf_counter()
    
    #########################################################################
    Um = forwardSolver_.forwardNSteps(
        u0_.clone(), u1_.clone(), gamma_, F_, Nx_, Ny_, N_, numberOfSources, device
    )[:, selx_, sely_, 1:]  # zeroth timestep removed
    #########################################################################
    #this part is the memory inefficient part that causes the problem. after computing the entire wavefield ~8 GiB, we only use the values of the sensor locations ~1.5 MiB. This is solved in the _forward_sensor_measurements function.
    
    
    gamma = torch.ones((1, 1, Nx + 3, Ny + 3)).to(device)
    gamma[:, :, 1:-1, 1:-1] = gamma_[:, :, 1:-1:factorToAvoidInverseCrime, 1:-1:factorToAvoidInverseCrime]
    Um = Um[:, :, (factorToAvoidInverseCrime - 1):: factorToAvoidInverseCrime]  # skip first small time steps

    # Compute gradient with adjoint method on intended grid
    _, gradient = AdjointMethod.getAdjointGradient(
        forwardSolver,
        u0.clone(), u1.clone(),
        c, rho, gamma * 0 + 1,
        F, Nx, dx, Ny, dy, N, dt,
        numberOfSources,
        Um, selx, sely, device)

    end = time.perf_counter()
    if case % 1 == 0:
        print(case)
        print("Elapased time: {:2f} ms".format((end - start) * 1000))

    # postprocessing
    gamma = gamma[0, 0, 1:-1, 1:-1].cpu().numpy()
    pd.DataFrame(gamma).to_hdf(
        destinationFolder + "/material" + str(case) + ".h5",
        key="gamma",
        index=False,
        mode="w",
        complevel=1,
    )

    pd.DataFrame([x0, y0, a, b, theta]).to_hdf(
        destinationFolder + "/parametersMaterial" + str(case) + ".h5",
        key="parameters",
        index=False,
        mode="w",
        complevel=1,
    )

    pd.DataFrame(Um.reshape(-1, N).cpu().numpy()).to_hdf(
        destinationFolder + "/measurement" + str(case) + ".h5",
        key="U",
        index=False,
        mode="w",
        complevel=1,
    )

    pd.DataFrame(gradient.cpu().numpy()).to_hdf(
        destinationFolder + "/gradient" + str(case) + ".h5",
        key="U",
        index=False,
        mode="w",
        complevel=1,
    )

pd.DataFrame(F.reshape(-1, N).cpu().numpy()).to_hdf(
    destinationFolder + "/source.h5", key="f", index=False, mode="w", complevel=1)
