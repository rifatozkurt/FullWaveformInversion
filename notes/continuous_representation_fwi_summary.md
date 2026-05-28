# Continuous Representation FWI Paper Notes

Source PDF:
`C:\Users\rozku\Desktop\Masters_Thesis\papers\UNVEILING THE MECHANISM OF CONTINUOUS REPRESENTATION FULL WAVEFORM INVERSION A WAVE BASED NEURAL TANGENT KERNEL FRAMEWORK.pdf`

Extracted text:
`notes/continuous_representation_fwi_paper.txt`

## Main Thesis

The paper frames continuous-representation FWI (CR-FWI) as FWI where the physical model is
represented by a coordinate network. Instead of directly optimizing one parameter per grid cell,
the inversion optimizes neural parameters. The wave equation remains in the loop, so the loss is
still the seismic-data misfit.

The paper's key theoretical claim is that CR-FWI is governed by a wave-based neural tangent
kernel (wave-based NTK): a composition of the physics sensitivity kernel and the representation
network's NTK. This architecture-dependent smoothing explains why INR-FWI is robust to poor
initial models but slow to recover high-frequency detail.

## Baseline IFWI / SIREN-INR

The paper's baseline INR method, called IFWI, uses a SIREN-style coordinate MLP:

- Input: spatial coordinate.
- Activation: sine activation.
- Frequency parameter: omega0 = 30.
- MLP depth: 4.
- Hidden width: 128.
- Optimizer: Adam.
- Learning rate for CR-FWI methods: 1e-4.

The network represents a velocity perturbation added to an initial velocity model:

`m_theta(x) = F_theta(x) + m0(x)`

This is different from an output-clamped material field unless the code explicitly maps the network
output to a perturbation around an initial model.

## LR-FWI

LR-FWI replaces the full 2D coordinate MLP with a low-rank tensor-function representation.
For a 2D model, the paper describes:

`F_theta(x1, x2) = F_theta1(x1) x C x F_theta2(x2)^T`

where:

- `F_theta1` is a 1D coordinate network mapping `x1` to a rank feature vector.
- `F_theta2` is a 1D coordinate network mapping `x2` to a rank feature vector.
- `C` is a learnable core matrix.
- Trainable parameters are `{theta1, theta2, C}`.

Paper hyperparameters:

- Same sine activation as IFWI.
- Rank is set to half of the model dimension.
- Each 1D MLP depth: 3.
- Hidden width: 128.

The intended effect is to encode low-rank/non-local similarity in the model. Empirically, the paper
claims LR-FWI gives a more balanced eigenvalue decay than plain INR, improving high-frequency
convergence while keeping robustness better than MPE.

## MPE-FWI

MPE-FWI uses a trainable multi-resolution hash-grid encoding before a small MLP.

Pipeline:

1. A coordinate is queried against multiple grid levels.
2. Features from nearby grid vertices are interpolated.
3. The multilevel feature vector is passed to a small MLP.
4. The MLP outputs the physical property value.

Paper hyperparameters:

- Multi-resolution hash grid.
- 16 levels.
- Base resolution: 50.
- Per-level scale factor: 1.05.
- Decoder MLP: 2 hidden layers.
- Hidden width: 64.

The intended effect is to reduce spectral bias and speed high-frequency convergence. The paper
argues that MPE has larger/elevated NTK eigenvalues compared with INR, so it learns detail faster.
However, it is less robust when the initial model is poor or data quality is degraded.

## IG-FWI Context

Although not requested as an implementation target here, the paper presents IG-FWI as the preferred
trade-off method. IG-FWI combines MPE features and a small INR feature network, then fuses them
with a compact MLP. It aims to place the eigenvalue decay between INR and MPE.

Paper hyperparameters:

- Hash grid: base resolution 50, 16 levels.
- Sine feature network: 2 layers, 128 neurons, omega0 = 30.
- Fusion MLP: 2 layers, 64 neurons.

## Comparison to This Repository

Current `INRSIREN` resembles the IFWI baseline in width, hidden layer count, sine activation, and
omega0. Important differences:

- The paper represents a perturbation around an initial model: `m_theta = F_theta + m0`.
- The repository maps sigmoid output to `[gamma0, 1]`: `output * (1 - gamma0) + gamma0`.
- The paper describes velocity inversion; this repository inverts the `gamma` field used by the
  local finite-difference solver.
- The paper's theory uses NTK-style scaling for a shallow example. The repository uses practical
  SIREN initialization instead.
- The repository uses adjoint-gradient injection into `gammaPred.grad`, not direct autograd through
  a PyTorch wave solver.
- The repository default epochs are much lower than many paper experiments, which often use
  hundreds of epochs and report 500 or 1500 epochs in difficult settings.

