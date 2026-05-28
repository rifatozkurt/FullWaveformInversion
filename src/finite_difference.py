import torch


class FiniteDifference(torch.jit.ScriptModule):
    def __init__(
            self,
            dt: float,
            dx: float,
            dy: float,
            c: float,
            rho: float,
            device: torch.device,
    ):
        super().__init__()

        dtcdx2 = torch.tensor((dt * c / dx) ** 2, dtype=torch.float32)
        dtcdy2 = torch.tensor((dt * c / dy) ** 2, dtype=torch.float32)
        self.dt2rho = torch.tensor(dt ** 2 / rho, dtype=torch.float32)

        # in x direction
        self.filtergammax1 = torch.nn.Conv2d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filtergammax1.weight = torch.nn.Parameter(
            torch.tensor([[[[0, 0, 0], [0, 0.5, 0], [0, 0.5, 0]]]], device=device),
            requires_grad=False,
        )
        self.filterux1 = torch.nn.Conv2d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filterux1.weight = torch.nn.Parameter(
            torch.tensor(
                [[[[0, 0, 0], [0, -dtcdx2, 0], [0, dtcdx2, 0]]]], device=device
            ),
            requires_grad=False,
        )

        self.filtergammax2 = torch.nn.Conv2d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filtergammax2.weight = torch.nn.Parameter(
            torch.tensor([[[[0, 0.5, 0], [0, 0.5, 0], [0, 0, 0]]]], device=device),
            requires_grad=False,
        )
        self.filterux2 = torch.nn.Conv2d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filterux2.weight = torch.nn.Parameter(
            torch.tensor(
                [[[[0, -dtcdx2, 0], [0, dtcdx2, 0], [0, 0, 0]]]], device=device
            ),
            requires_grad=False,
        )

        # in y direction
        self.filtergammay1 = torch.nn.Conv2d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filtergammay1.weight = torch.nn.Parameter(
            torch.tensor([[[[0, 0, 0], [0, 0.5, 0.5], [0, 0, 0]]]], device=device),
            requires_grad=False,
        )
        self.filteruy1 = torch.nn.Conv2d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filteruy1.weight = torch.nn.Parameter(
            torch.tensor(
                [[[[0, 0, 0], [0, -dtcdy2, dtcdy2], [0, 0, 0]]]], device=device
            ),
            requires_grad=False,
        )

        self.filtergammay2 = torch.nn.Conv2d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filtergammay2.weight = torch.nn.Parameter(
            torch.tensor([[[[0, 0, 0], [0.5, 0.5, 0], [0, 0, 0]]]], device=device),
            requires_grad=False,
        )
        self.filteruy2 = torch.nn.Conv2d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filteruy2.weight = torch.nn.Parameter(
            torch.tensor(
                [[[[0, 0, 0], [-dtcdy2, dtcdy2, 0], [0, 0, 0]]]], device=device
            ),
            requires_grad=False,
        )

    def forward(self, u0, u1, gamma, f):
        gammainv = 1.0 / gamma
        u2 = (
                -u0
                + 2 * u1
                + (
                        1.0 / self.filtergammax1(gammainv) * self.filterux1(u1)
                        - 1.0 / self.filtergammax2(gammainv) * self.filterux2(u1)
                        + 1.0 / self.filtergammay1(gammainv) * self.filteruy1(u1)
                        - 1.0 / self.filtergammay2(gammainv) * self.filteruy2(u1)
                        + self.dt2rho * f
                )
                / gamma
        )

        # ghost cells
        u2[:, :, 0, :] = u2[:, :, 2, :]
        u2[:, :, -1, :] = u2[:, :, -3, :]
        u2[:, :, :, 0] = u2[:, :, :, 2]
        u2[:, :, :, -1] = u2[:, :, :, -3]

        return u2

    def forwardNSteps(
            self,
            u0,
            u1,
            gamma,
            f,
            Nx: int,
            Ny: int,
            N: int,
            numberOfSimulations: int,
            device: torch.device,
    ):
        U = torch.zeros((numberOfSimulations, Nx + 3, Ny + 3, N + 1), device=device)

        for i in range(N):
            u2 = self.forward(u0.clone(), u1.clone(), gamma, f[:, :, :, :, i])
            u0[:], u1[:] = u1, u2
            U[:, :, :, i + 1] = u2[:, 0]
        return U


class FiniteDifference3D(torch.jit.ScriptModule):
    def __init__(
            self,
            dt: float,
            dx: float,
            dy: float,
            dz: float,
            c: float,
            rho: float,
            device: torch.device,
    ):
        super().__init__()

        dtcdx2 = torch.tensor((dt * c / dx) ** 2, dtype=torch.float32)
        dtcdy2 = torch.tensor((dt * c / dy) ** 2, dtype=torch.float32)
        dtcdz2 = torch.tensor((dt * c / dz) ** 2, dtype=torch.float32)
        self.dt2rho = torch.tensor(dt ** 2 / rho, dtype=torch.float32)

        # in x direction
        self.filtergammax1 = torch.nn.Conv3d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filtergammax1.weight.data *= 0.0
        self.filtergammax1.weight.data[0, 0, :, 1, 1] = torch.tensor(
            [0.0, 0.5, 0.5], device=device
        )
        self.filtergammax1.weight.requires_grad = False

        self.filterux1 = torch.nn.Conv3d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filterux1.weight.data *= 0.0
        self.filterux1.weight.data[0, 0, :, 1, 1] = torch.tensor(
            [0.0, -dtcdx2, dtcdx2], device=device
        )
        self.filterux1.weight.requires_grad = False

        self.filtergammax2 = torch.nn.Conv3d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filtergammax2.weight.data *= 0.0
        self.filtergammax2.weight.data[0, 0, :, 1, 1] = torch.tensor(
            [0.5, 0.5, 0.0], device=device
        )
        self.filtergammax2.weight.requires_grad = False

        self.filterux2 = torch.nn.Conv3d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filterux2.weight.data *= 0.0
        self.filterux2.weight.data[0, 0, :, 1, 1] = torch.tensor(
            [-dtcdx2, dtcdx2, 0], device=device
        )
        self.filterux2.weight.requires_grad = False

        # in y direction
        self.filtergammay1 = torch.nn.Conv3d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filtergammay1.weight.data *= 0.0
        self.filtergammay1.weight.data[0, 0, 1, :, 1] = torch.tensor(
            [0.0, 0.5, 0.5], device=device
        )
        self.filtergammay1.weight.requires_grad = False

        self.filteruy1 = torch.nn.Conv3d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filteruy1.weight.data *= 0.0
        self.filteruy1.weight.data[0, 0, 1, :, 1] = torch.tensor(
            [0.0, -dtcdy2, dtcdy2], device=device
        )
        self.filteruy1.weight.requires_grad = False

        self.filtergammay2 = torch.nn.Conv3d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filtergammay2.weight.data *= 0.0
        self.filtergammay2.weight.data[0, 0, 1, :, 1] = torch.tensor(
            [0.5, 0.5, 0.0], device=device
        )
        self.filtergammay2.weight.requires_grad = False

        self.filteruy2 = torch.nn.Conv3d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filteruy2.weight.data *= 0.0
        self.filteruy2.weight.data[0, 0, 1, :, 1] = torch.tensor(
            [-dtcdy2, dtcdy2, 0], device=device
        )
        self.filteruy2.weight.requires_grad = False

        # in z direction
        self.filtergammaz1 = torch.nn.Conv3d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filtergammaz1.weight.data *= 0.0
        self.filtergammaz1.weight.data[0, 0, 1, 1, :] = torch.tensor(
            [0.0, 0.5, 0.5], device=device
        )
        self.filtergammaz1.weight.requires_grad = False

        self.filteruz1 = torch.nn.Conv3d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filteruz1.weight.data *= 0.0
        self.filteruz1.weight.data[0, 0, 1, 1, :] = torch.tensor(
            [0.0, -dtcdz2, dtcdz2], device=device
        )
        self.filteruz1.weight.requires_grad = False

        self.filtergammaz2 = torch.nn.Conv3d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filtergammaz2.weight.data *= 0.0
        self.filtergammaz2.weight.data[0, 0, 1, 1, :] = torch.tensor(
            [0.5, 0.5, 0.0], device=device
        )
        self.filtergammaz2.weight.requires_grad = False

        self.filteruz2 = torch.nn.Conv3d(
            1,
            1,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            padding_mode="zeros",
            device=device,
        )
        self.filteruz2.weight.data *= 0.0
        self.filteruz2.weight.data[0, 0, 1, 1, :] = torch.tensor(
            [-dtcdz2, dtcdz2, 0], device=device
        )
        self.filteruz2.weight.requires_grad = False

    @torch.jit.script_method
    def forward(self, u0, u1, gamma, f):
        gammainv = 1.0 / gamma  # should maybe be input?

        u2 = (
                -u0
                + 2 * u1
                + (
                        1.0 / self.filtergammax1(gammainv) * self.filterux1(u1)
                        - 1.0 / self.filtergammax2(gammainv) * self.filterux2(u1)
                        + 1.0 / self.filtergammay1(gammainv) * self.filteruy1(u1)
                        - 1.0 / self.filtergammay2(gammainv) * self.filteruy2(u1)
                        + 1.0 / self.filtergammaz1(gammainv) * self.filteruz1(u1)
                        - 1.0 / self.filtergammaz2(gammainv) * self.filteruz2(u1)
                        + self.dt2rho * f
                )
                / gamma
        )

        # ghost cells
        u2[:, :, 0, :, :] = u2[:, :, 2, :, :]
        u2[:, :, -1, :, :] = u2[:, :, -3, :]
        u2[:, :, :, 0, :] = u2[:, :, :, 2, :]
        u2[:, :, :, -1, :] = u2[:, :, :, -3, :]
        u2[:, :, :, :, 0] = u2[:, :, :, :, 2]
        u2[:, :, :, :, -1] = u2[:, :, :, :, -3]

        return u2

    def forwardNSteps(
            self, u0, u1, gamma, f, Nx, Ny, Nz, N, dt, numberOfSimulations, device
    ):
        U = torch.zeros(
            (numberOfSimulations, Nx + 3, Ny + 3, Nz + 3, N + 1), device=device
        )
        t = torch.linspace(0, (N - 1) * dt, N)

        for i in range(N):
            u2 = self.forward(u0.clone(), u1.clone(), gamma, f(t[i]).unsqueeze(0))
            u0[:], u1[:] = u1, u2
            U[:, :, :, :, i + 1] = u2[
                                   :, 0
                                   ]
        return U
