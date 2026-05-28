import torch

# gradient computation
def getDerivativeInT(u, dt):
    dudt = (u[:, :, :, 2:] - u[:, :, :, :-2]) / 2.0 / dt
    return dudt


def getDerivativeInX(u, dx):
    dudx = (u[:, 2:, :, :] - u[:, :-2, :, :]) / 2.0 / dx
    return dudx


def getDerivativeInY(u, dy):
    dudy = (u[:, :, 2:, :] - u[:, :, :-2, :]) / 2.0 / dy
    return dudy


def getAdjointGradient(
    ForwardSolver,
    u0,
    u1,
    c,
    rho,
    gamma,
    f,
    Nx,
    dx,
    Ny,
    dy,
    N,
    dt,
    numberOfSources,
    um,
    selx,
    sely,
    device,
):
    # forward problem
    U = ForwardSolver.forwardNSteps(
        u0.clone(), u1.clone(), gamma, f, Nx, Ny, N, numberOfSources, device)


    # cost function
    cost = (
        0.5* torch.sum(
            dx * dy * torch.trapz((U[:, selx, sely, 1:] - um) ** 2, dx=dt, axis=2))/ numberOfSources)


    # adjoint problem
    fadjoint = torch.zeros((numberOfSources, 1, Nx + 3, Ny + 3, N), device=device)
    fadjoint[:, 0, selx, sely, :] = -(U[:, selx, sely,1:] - um)  # [:,:,1:] # removal of initial conditions
    fadjoint = torch.flip(fadjoint, dims=(4,))
    Uadjoint = ForwardSolver.forwardNSteps(
        u0.clone(), u1.clone(), gamma, fadjoint, Nx, Ny, N, numberOfSources, device)
    Uadjoint = torch.flip(Uadjoint, dims=(3,))

    integrand = -rho * (getDerivativeInT(Uadjoint, dt) * getDerivativeInT(U, dt))[
        :, 1:-1, 1:-1, :
    ] + rho * c**2 * (
        (getDerivativeInX(Uadjoint, dx) * getDerivativeInX(U, dx))[:, :, 1:-1, 1:-1]
        + (getDerivativeInY(Uadjoint, dy) * getDerivativeInY(U, dy))[:, 1:-1, :, 1:-1]
    )

    kernel = torch.mean(torch.trapz(integrand, dx=dt, axis=3), axis=0)

    gradient = (
        kernel * dx * dy
    )  # spatial integration of kernel (diracs as shape functions)

    gradient *= (
        2
    )

    return cost, gradient


def getDerivativeInT3D(u, dt):
    dudt = (u[:, :, :, :, 2:] - u[:, :, :, :, :-2]) / 2.0 / dt
    return dudt


def getDerivativeInX3D(u, dx):
    dudx = (u[:, 2:, :, :, :] - u[:, :-2, :, :, :]) / 2.0 / dx
    return dudx


def getDerivativeInY3D(u, dy):
    dudy = (u[:, :, 2:, :, :] - u[:, :, :-2, :, :]) / 2.0 / dy
    return dudy


def getDerivativeInZ3D(u, dz):
    dudy = (u[:, :, :, 2:, :] - u[:, :, :, :-2, :]) / 2.0 / dz
    return dudy


def getAdjointGradient3D(
    ForwardSolver,
    u0,
    u1,
    c,
    rho,
    gamma,
    f,
    Nx,
    dx,
    Ny,
    dy,
    Nz,
    dz,
    N,
    dt,
    numberOfSources,
    um,
    selx,
    sely,
    selz,
    device,
):
    # forward problem
    U = ForwardSolver.forwardNSteps(
        u0.clone(), u1.clone(), gamma, f, Nx, Ny, Nz, N, dt, numberOfSources, device
    )

    # cost function
    cost = (
        0.5
        * dx
        * dy
        * dz
        * torch.sum(torch.trapz((U[:, selx, sely, selz, :] - um) ** 2, dx=dt, axis=2))
        / numberOfSources
    )

    # adjoint problem
    fadjoint_ = torch.zeros(
        (numberOfSources, 1, Nx + 3, Ny + 3, Nz + 3, N), device=device
    )
    fadjoint_[:, 0, selx, sely, selz, :] = -(U[:, selx, sely, selz, :] - um)[:, :, 1:]
    fadjoint_ = torch.flip(fadjoint_, dims=(5,))

    def fadjoint(t):
        # convert timestep to iteration number
        timestep = int(t / dt)
        return fadjoint_[:, 0, :, :, :, timestep]

    Uadjoint = ForwardSolver.forwardNSteps(
        u0.clone(),
        u1.clone(),
        gamma,
        fadjoint,
        Nx,
        Ny,
        Nz,
        N,
        dt,
        numberOfSources,
        device,
    )
    Uadjoint = torch.flip(Uadjoint, dims=(4,))

    integrand = -rho * (getDerivativeInT3D(Uadjoint, dt) * getDerivativeInT3D(U, dt))[
        :, 1:-1, 1:-1, 1:-1, :
    ] + rho * c**2 * (
        (getDerivativeInX3D(Uadjoint, dx) * getDerivativeInX3D(U, dx))[
            :, :, 1:-1, 1:-1, 1:-1
        ]
        + (getDerivativeInY3D(Uadjoint, dy) * getDerivativeInY3D(U, dy))[
            :, 1:-1, :, 1:-1, 1:-1
        ]
        + (getDerivativeInZ3D(Uadjoint, dz) * getDerivativeInZ3D(U, dz))[
            :, 1:-1, 1:-1, :, 1:-1
        ]
    )

    kernel = torch.mean(torch.trapz(integrand, dx=dt, axis=4), axis=0)

    gradient = kernel * dx * dy * dz

    gradient *= 2

    return cost, gradient
