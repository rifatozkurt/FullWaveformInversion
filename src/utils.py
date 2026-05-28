import numpy as np
import torch


def diracx(x, dx, i):
    x = x * 0
    x[i, :] = 1 / dx
    return x


def diracy(y, dy, j):
    y = y * 0
    y[:, j] = 1 / dy
    return y


def generateSineBurst(frequency, cycles, amplitude):
    omega = frequency * 2 * np.pi
    return (
        lambda t: amplitude
                  * ((t <= cycles / frequency) & (t > 0))
                  * np.sin(omega * t)
                  * (np.sin(omega * t / 2 / cycles)) ** 2
    )  # normalization over the applied area


def generateCoseBurst(frequency, cycles, amplitude):
    omega = frequency * 2 * np.pi
    return (lambda t: amplitude
                      * ((t <= cycles / frequency) & (t > 0))
                      * np.cos(omega * t)
                      * (np.sin(omega * t / 2 / cycles)) ** 2)  # normalization over the applied area


def generateCose2Burst(frequency, cycles, amplitude):
    omega = frequency * 2 * np.pi
    return lambda t: amplitude * ((t <= cycles / frequency) & (t > 0)) * [
        np.cos(omega * 1.0 * t) * np.sin(omega * t / 2 / cycles) + np.sin(omega * 0.9 * t) * np.sin(
            omega * t / 2 / cycles)]  # normalization over the applied area


def generatesinc(frequency, cycles, amplitude):
    omega = frequency * 2 * np.pi
    return lambda t: amplitude * ((t <= 2 * cycles / frequency) & (t > 0)) * np.sin(0.4 * omega * t) * (
        np.sin(0.5 * omega * t / 2 / cycles)) ** 2


def getSource(
        frequency,
        cycles,
        amplitude,
        sourceLocationsx,
        sourceLocationsy,
        Nx,
        Ny,
        dx,
        dy,
        Lx,
        Ly,
        N,
        dt,
):
    sourcesin = generateSineBurst(frequency, cycles, amplitude)

    f_sourcesin = lambda x, y, t, i, j: diracx(x, dx, i) * diracy(y, dy, j) * sourcesin(t)

    F = torch.zeros((len(sourceLocationsx), 1, Nx + 3, Ny + 3, N))
    # Generate the spatial grid
    x = np.linspace(0 - dx, Lx + dx, Nx + 3)  # with ghost cells
    y = np.linspace(0 - dy, Ly + dy, Ny + 3)  # with ghost cells
    # Generate the temporal grid
    t = np.linspace(0, (N - 1) * dt, N)
    y, x, t = np.meshgrid(y, x, t)
    for iSource in range(len(sourceLocationsx)):
        if iSource == 0:
            F[iSource] = torch.from_numpy(
                f_sourcesin(x, y, t, sourceLocationsx[iSource], sourceLocationsy[iSource])).unsqueeze(0)
        if iSource == 1:
            F[iSource] = torch.from_numpy(
                f_sourcesin(x, y, t, sourceLocationsx[iSource], sourceLocationsy[iSource])).unsqueeze(0)
        if iSource == 2:
            F[iSource] = torch.from_numpy(
                f_sourcesin(x, y, t, sourceLocationsx[iSource], sourceLocationsy[iSource])).unsqueeze(0)
        else:
            F[iSource] = torch.from_numpy(
                f_sourcesin(x, y, t, sourceLocationsx[iSource], sourceLocationsy[iSource])).unsqueeze(0)

    return F


def getSourceDirect(
        frequency,
        cycles,
        amplitude,
        sourceLocationsx,
        sourceLocationsy,
        Nx,
        Ny,
        dx,
        dy,
        N,
        dt,
):
    # Memory-patched alternative to getSource(...). It returns the same dense
    # source tensor F, but avoids creating dense x/y/t meshgrid temporaries.
    sourcesin = generateSineBurst(frequency, cycles, amplitude)
    t = np.linspace(0, (N - 1) * dt, N)
    
    #this line basically handles the dirac functions by only computing the source time trace once and then assigning it to the correct locations in the source tensor F. This avoids creating large intermediate arrays for x, y, and t, which can be memory-intensive.
    source_time = torch.as_tensor(sourcesin(t), dtype=torch.float32) / dx / dy
    F = torch.zeros((len(sourceLocationsx), 1, Nx + 3, Ny + 3, N), dtype=torch.float32)
    for iSource in range(len(sourceLocationsx)):
        F[iSource, 0, sourceLocationsx[iSource], sourceLocationsy[iSource], :] = source_time
    return F


def getSourceTime(frequency, cycles, amplitude, N, dt, dx, dy, device):
    # this generates time trace of the source step by step so that _forward_sensor_measurements can use it step by step to calculate the wavefield for the full time trace without having to store it all.
    sourcesin = generateSineBurst(frequency, cycles, amplitude)
    t = np.linspace(0, (N - 1) * dt, N)
    source_time = torch.as_tensor(sourcesin(t), dtype=torch.float32, device=device)
    return source_time / dx / dy


def generateGamma(Nx, Ny, dx, dy, Lx, Ly, gamma0, boundaryFactor=1):
    # Generate the spatial grid
    x = np.linspace(0 - dx, Lx + dx, Nx + 3)  # with ghost cells
    y = np.linspace(0 - dy, Ly + dy, Ny + 3)  # with ghost cells
    x, y = np.meshgrid(x, y, indexing="ij")

    gamma = x * 0 + 1

    isInsideBoundary = False
    while isInsideBoundary is not True:
        a = (
                np.random.rand() * (Ly - 4 * dy) * 0.5 + 2 * dy
        )  # maximal size is half length in y direction (minimum at least 2*dy, with 2*dy as boundary distance)
        b = np.random.rand() * (0.2 * Ly - 4 * dy) * 0.5 + 2 * dy
        theta = np.random.rand() * np.pi
        x0 = np.random.rand() * (Lx - 2 * dx) + dx  # ensure large enough range
        y0 = np.random.rand() * (Ly - 2 * dy) + dy

        index = (
                        ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)) ** 2 / a ** 2
                        + ((x - x0) * np.sin(theta) - (y - y0) * np.cos(theta)) ** 2 / b ** 2
                ) < 1

        boundary = [i for i in range(1 + boundaryFactor)]
        boundary += [-(i + 1) for i in range(1 + boundaryFactor)]
        if np.any(index[boundary, :]) == False and np.any(index[:, boundary]) == False:
            isInsideBoundary = True

    gamma[index] = gamma0
    return (torch.from_numpy(gamma).unsqueeze(0).unsqueeze(0).to(torch.float32),
            x0,
            y0,
            a,
            b,
            theta)


def getSourceLocations(Nx, Ny, distanceBetweenSources, numberOfSources):
    sourceLocationsx = [
        (Nx + 1) // 2 - i * distanceBetweenSources
        for i in reversed(range(1, numberOfSources // 2 + 1))
    ]
    sourceLocationsx += [
        (Nx + 1) // 2 + i * distanceBetweenSources
        for i in range(1, numberOfSources // 2 + 1)
    ]
    sourceLocationsy = [Ny + 1] * len(sourceLocationsx)
    return sourceLocationsx, sourceLocationsy


def getSensorLocations(Nx, Ny, distanceBetweenSensors, sourceLocationsx):
    bound = int(np.ceil((sourceLocationsx[-1] - (Nx + 1) / 2) / distanceBetweenSensors))
    selx = [
        (Nx + 1) // 2 - i * distanceBetweenSensors
        for i in reversed(range(1, bound + 1))
    ]
    selx += [(Nx + 1) // 2 + i * distanceBetweenSensors for i in range(1, bound + 1)]
    sely = [Ny + 1] * len(selx)
    return selx, sely


def convertStringToList(string):
    string = string.strip("[]")
    string = string.split(",")
    string = [int(element) for element in string]
    return string
