# Transfer Learning FWI Research Code

This repository is a light restructure of the original full waveform inversion scripts. The numerical routines are kept close to the legacy code; the main changes are folder organization, imports, and YAML configuration.

## Layout

- `legacy/`: original files, left untouched
- `src/`: copied core modules and script-like experiment functions
- `scripts/`: thin command-line wrappers
- `configs/default.yaml`: simulation settings and experiment hyperparameters
- `data/`, `models/`, `runs/`: generated data, model weights, and run outputs

## Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run from the repository root:

```powershell
python scripts/generate_train_data.py --config configs/default.yaml
python scripts/generate_test_data.py --config configs/default.yaml
python scripts/pretrain.py --config configs/default.yaml
python scripts/run_experiment.py --method conventional_fwi --case 1 --config configs/default.yaml
python scripts/run_all_experiments.py --config configs/default.yaml
python scripts/run_all_methods.py --config configs/default.yaml
```

The data-generation wrappers call `src.data_generation.generate_dataset(...)`.
The pretraining wrapper calls `src.pretraining.pretrain_unet(...)` and writes model weights to `models/pretrained/`.
The default pretraining config uses 800 samples and 100 epochs, matching the pretrained model naming expected by the transfer-learning experiments.
Pretraining runs are saved under `runs/pretraining/` with `config.yaml`, `runtime.txt`, loss histories, sample/split indices, and a training/validation loss plot.

## Gradio UI

Launch the first UI foundation from the repository root:

```powershell
python scripts/launch_gradio.py
```

Direct execution also works:

```powershell
python src/ui/app.py
```

The UI supports:

- Tab 1: run existing FWI experiments
- Tab 2: view existing cases, split `data/raw/` into train/test folders, and create custom canvas cases
- Tab 3: pretrain U-Net models

Canvas-generated custom cases are saved under `data/custom/` by default. When measurement/gradient generation is enabled, the canvas gamma is lifted to the finer reference grid and downsampled back to the inversion grid, matching the train/test data-generation convention.

The raw-data split tool copies `material*.h5`, `gradient*.h5`, optional `measurement*.h5`, optional `parametersMaterial*.h5`, and `source.h5` from `data/raw/` into contiguous `data/train/` and `data/test/` case numbering.

The experiment tab includes a Python output box that captures printed training/status messages after each UI-triggered run finishes.

Experiment outputs are saved under timestamped folders in `runs/`. Each run folder contains:

- `config.yaml`
- `runtime.txt`
- `figures/`
- `histories/`
- `outputs/`

## Parameter Reference

Most parameters live in `configs/default.yaml`. The debug configs keep the same structure but use smaller or shorter runs.

### Paths

- `paths.train_data`: folder used by pretraining, usually `data/train`.
- `paths.test_data`: default generated test-data folder.
- `paths.casestudy_data`: folder for case-study data.
- `paths.pretrained_models`: where pretrained U-Net weights are saved and loaded from.
- `paths.checkpoints`: optional checkpoint folder.
- `paths.runs`: root folder for experiment and pretraining run outputs.

### Physical and Numerical Simulation

- `simulation.Lx`, `simulation.Ly`: physical domain size in the x and y directions.
- `simulation.Nx`, `simulation.Ny`: number of grid intervals in x and y, excluding ghost cells. Material fields use shape `(Nx + 1, Ny + 1)`, while solver tensors include ghost cells.
- `simulation.dt`: time-step size for the inversion-grid finite-difference solver.
- `simulation.N`: number of time steps on the inversion grid.
- `simulation.gamma0`: lower material value used for void/damage regions. Intact material is represented by `gamma = 1`.
- `simulation.rho`: density used in the wave equation and adjoint gradient.
- `simulation.c`: wave speed used in the finite-difference solver.
- `simulation.numberOfSources`: number of source locations.
- `simulation.distanceBetweenSources`: source spacing in grid-index units.
- `simulation.distanceBetweenSensors`: receiver/sensor spacing in grid-index units.

### Source Signal

- `source.frequency`: sine-burst frequency.
- `source.cycles`: number of cycles in the sine burst.
- `source.amplitude`: source amplitude before spatial normalization by `dx * dy`.

### Data Generation

- `data_generation.seed`: PyTorch random seed used before generated train/test datasets.
- `data_generation.train.number_of_cases`: number of generated training cases.
- `data_generation.train.number_of_damages`: number of random damage inclusions per generated training case.
- `data_generation.train.factor_to_avoid_inverse_crime`: fine-grid factor for generated training data. A value of `2` means the forward measurement solve uses `2*Nx`, `2*Ny`, `2*N` and is downsampled afterward.
- `data_generation.test.*`: same fields for generated test data.

The Gradio raw-data split tool does not regenerate physics. It copies existing numeric cases from `data/raw/` into contiguous `data/train/` and `data/test/` numbering using the selected split ratio and seed.

### U-Net Architecture

- `models.unet.channels`: encoder/decoder channel widths for `Unet`, for example `[1, 16, 32, 64, 128]`.
- `models.unet.number_of_convolutions_per_block`: number of convolution layers per U-Net block.
- `models.unet.batch_norm`: whether the U-Net blocks use batch normalization.

The legacy architecture is kept in `src/networks.py`; these fields only choose the same constructor arguments used by the original scripts.

### Pretraining

- `pretraining.seed`: Torch seed used before model initialization.
- `pretraining.split_seed`: seed for the internal 80/20 train/validation split.
- `pretraining.numberOfSamples`: number of cases loaded from the pretraining data directory.
- `pretraining.availableSamples`: size of the index pool sampled from. The code samples from `0..availableSamples-1`.
- `pretraining.trainingType`: label included in saved model names, usually `supervised`.
- `pretraining.model_type`: model name label, usually `Unet`.
- `pretraining.NNchannels`: legacy channel list. The active UI/backend also mirrors this into `models.unet.channels`.
- `pretraining.numberOfConvolutionsPerBlock`: legacy convolution-block count. Mirrored into `models.unet`.
- `pretraining.batchDivisor`: batch size is computed as `numberOfSamples // batchDivisor`.
- `pretraining.lr`: RMSprop learning rate.
- `pretraining.alpha`, `pretraining.beta`: learning-rate scheduler parameters for `(beta * epoch + 1) ** alpha`. For legacy compatibility, `beta` is reset to zero before the loop in `pretrain_unet`, matching `legacy/Pretraining.py`.
- `pretraining.epochs`: number of pretraining epochs. The default is `100` to match the pretrained model expected by transfer-learning experiments.
- `pretraining.clipGrad`: gradient-clipping norm.
- `pretraining.l2`: legacy field retained in config. The supervised legacy pretraining loop does not use it directly.
- `pretraining.costScaling`: legacy field retained in config. The supervised legacy pretraining loop does not use it directly.

Pretraining validation is an internal holdout from the selected training folder: 80% of `numberOfSamples` is used for training and 20% for validation. The external `data/test` split is used later for FWI experiments, not for pretraining validation.

### Experiment Defaults

Shared experiment fields:

- `experiments.cases`: default case IDs for batch scripts.
- `experiments.pretrain_samples`: list of pretrained sample counts used by transfer-learning style experiments.
- `experiments.epochs_pretrain`: epoch count used to construct pretrained model filenames, for example `model_Unet_100_supervised_800_channel_5`.
- `experiments.modelType`, `experiments.NNchannels`, `experiments.numberOfConvolutionsPerBlock`: model-name and architecture fields used when loading pretrained U-Net models.

Common optimizer fields inside each method:

- `lr`: learning rate.
- `alpha`, `beta`: scheduler parameters for `(beta * epoch + 1) ** alpha`.
- `epochs`: number of inversion/training epochs for that method.
- `clipGrad`: gradient-clipping norm.
- `costScaling`: scale factor applied to adjoint gradients before optimizer steps.
- `l2`: weight decay where the method uses neural-network parameters.
- `weightLrFactor`: legacy field for conventional methods; currently preserved for compatibility with the original settings.

Method-specific fields:

- `experiments.conventional_fwi`: optimizes the material field directly from homogeneous initialization.
- `experiments.conventional_fwi_initial_guess.seed`: seed used before loading the pretrained initial guess.
- `experiments.conventional_fwi_initial_guess.pretrain_samples`: sample count used in the pretrained model filename for the initial guess.
- `experiments.nn_based_fwi.seed`: seed used before initializing the neural parameterization.
- `experiments.nn_based_fwi.input_shape`: shape of the learned low-dimensional CNN input tensor.
- `experiments.nn_based_fwi.trunc_normal_mean`, `trunc_normal_std`, `trunc_normal_a`, `trunc_normal_b`: truncated-normal initialization parameters for the NN-based input tensor.
- `experiments.inr_fwi.hidden_features`, `hidden_layers`: width and depth of the coordinate MLP used for basic INR-FWI. The INR receives normalized `(x, y)` coordinates and outputs one material value per grid point.
- `experiments.inr_siren_fwi.hidden_features`, `hidden_layers`, `omega0`: width, depth, and sine frequency for SIREN-style INR-FWI. This method keeps the repo's gamma scaling while using sine activations and SIREN initialization.
- `experiments.transfer_learning_fwi.seed`: seed used before fine-tuning the pretrained U-Net.
- `experiments.transfer_learning_fwi_frozen_encoder`: transfer-learning variant reserved for freezing the U-Net encoder and fine-tuning the decoder. The current placeholder `freeze_unet_encoder(...)` returns all parameters, so it behaves like normal transfer learning until freeze rules are filled in.

Quick debug data generation:

```powershell
python scripts/generate_train_data.py --config configs/quick_debug.yaml
python scripts/generate_test_data.py --config configs/quick_debug.yaml
```

To call the single-case helper directly:

```powershell
python -c "from src.config import load_config; from src.data_generation import generate_single_case; c=load_config('configs/quick_debug.yaml'); generate_single_case(c, 0, 'data/debug_single', n_damages=1, factor_to_avoid_inverse_crime=c['data_generation']['train']['factor_to_avoid_inverse_crime'])"
```

`quick_debug.yaml` uses a smaller grid and fewer time steps than the paper/default config, so it is intended for checking the data-writing pipeline rather than producing comparable research results.

Real-size timing debug data generation:

```powershell
python scripts/generate_train_data.py --config configs/normal_debug.yaml
python scripts/generate_test_data.py --config configs/normal_debug.yaml
```

`normal_debug.yaml` uses the default/full grid and time settings, but only one train case and one test case.

The included `pyproject.toml` lets Python import `src` from the repository root. If you run scripts from elsewhere, install editable mode with `pip install -e .` or set `PYTHONPATH` to the repository root.
