"""
Generate the three Colab notebooks, one per session.

    python scripts/make_colab_notebooks.py

Writes colab_session1_inr_freezing.ipynb, colab_session2_pretraining.ipynb and
colab_session3_evaluation.ipynb at the repo root. Regenerate rather than editing
the .ipynb files by hand so the three stay consistent.

Session split, and why:
  1  Self-contained: trains its own 800-sample U-Net, then the freezing ablation,
     then the INR experiments (which dominate the runtime and need no pretraining).
  2  The long pole: every model family at every sample count, in ONE run
     directory so the scaling curves are plotted in place.
  3  Depends on session 2's checkpoints, so it must run afterwards.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BRANCH = "improve_transformer"
GIT = "https://github.com/rifatozkurt/FullWaveformInversion"


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def preamble(title, blurb, needs_models=False):
    cells = [
        md(f"# {title}\n\n{blurb}"),
        md("## 1. Setup"),
        code(f"!git clone -b {BRANCH} {GIT}\n%cd FullWaveformInversion"),
        code("!pip install -q -r requirements.txt"),
        code(
            "import torch\n"
            "print('CUDA:', torch.cuda.is_available())\n"
            "if torch.cuda.is_available():\n"
            "    p = torch.cuda.get_device_properties(0)\n"
            "    print(f'{p.name}, {p.total_memory/1e9:.1f} GB')\n"
            "    # A single adjoint evaluation allocates ~3 GB. Anything under ~8 GB\n"
            "    # means running one job at a time and nothing else on the GPU.\n"
        ),
        code(
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            "OUT = '/content/drive/MyDrive/fwi_thesis'\n"
            "!mkdir -p {OUT}"
        ),
        md("## 2. Data\n\n"
           "`extended/` (ids 0-14999) is training data; `eval/` (ids 15000-15999) is\n"
           "held out. They do NOT overlap. Restore both from Drive if you have them\n"
           "zipped there, otherwise generate (slow).")
    ]
    restore = (
        "# Restore from Drive (fast path)\n"
        "!unzip -q -o {OUT}/data/extended.zip -d /content/  || echo 'no extended.zip'\n"
        "!unzip -q -o {OUT}/data/eval.zip     -d /content/  || echo 'no eval.zip'\n"
        "!ls /content/extended | head -3 ; ls /content/eval | head -3\n"
        "\n"
        "# --- alternative: generate instead (hours) ---\n"
        "# !python scripts/generate_train_data_colab.py \\\n"
        "#     --config configs/config_final.yaml --output-dir /content/extended \\\n"
        "#     --start-case-id 0 --number-of-cases 15000 --case-batch-size 4 --no-overwrite\n"
    )
    cells.append(code(restore))
    if needs_models:
        cells.append(md("### Restore session 2's checkpoints\n\n"
                        "This session cannot run without them."))
        cells.append(code(
            "!mkdir -p models\n"
            "!cp -r {OUT}/session2/models/final models/ 2>/dev/null || "
            "unzip -q -o {OUT}/session2/models.zip -d .\n"
            "!ls models/final | head -20\n"
            "import pathlib\n"
            "n = len(list(pathlib.Path('models/final').glob('*.pt'))) + \\\n"
            "    len(list(pathlib.Path('models/final').glob('model_Unet*')))\n"
            "print(f'{n} checkpoints found')\n"
            "assert n > 0, 'No checkpoints restored -- session 3 cannot run.'"
        ))
    return cells


def save_cells(out_name, run_label):
    return [
        md("## 4. Save everything to Drive\n\n"
           "`runs/` holds every history, CSV and figure; `models/` holds the\n"
           "checkpoints. Zip both so a disconnect does not lose the session."),
        code(
            f"SESSION = '{out_name}'\n"
            "!mkdir -p {OUT}/{SESSION}\n"
            "!zip -qr /content/runs.zip runs\n"
            "!cp /content/runs.zip {OUT}/{SESSION}/runs.zip\n"
            "!zip -qr /content/models.zip models\n"
            "!cp /content/models.zip {OUT}/{SESSION}/models.zip\n"
            "print('saved to', OUT + '/' + SESSION)"
        ),
        md("## 5. Look at the figures before you disconnect"),
        code(
            "from IPython.display import Image, display\n"
            "import pathlib\n"
            f"for p in sorted(pathlib.Path('{run_label}').rglob('report/*.png')):\n"
            "    print(p)\n"
            "    display(Image(str(p)))"
        ),
    ]


# --------------------------------------------------------------------------- #
def session1():
    cells = preamble(
        "Session 1 — U-Net pretraining, freezing ablation, INR experiments",
        "Self-contained: depends on no other session. The INR runs dominate the\n"
        "runtime. Expect several hours; every step is resumable, so re-running a\n"
        "cell after a disconnect skips what already finished.")
    cells += [
        md("## 3a. Pretrain the 800-sample U-Net\n\n"
           "This is the baseline the freezing ablation fine-tunes. 800 matches\n"
           "Singh et al., and keeps this session independent of session 2.\n"
           "Retraining is required regardless: the gamma-MSE definition, the input\n"
           "normalization and the decoder BatchNorms all changed."),
        code("!python scripts/pretrain.py \\\n"
             "    --config configs/config_final.yaml \\\n"
             "    --data-dir /content/extended \\\n"
             "    --output-dir models/final \\\n"
             "    --run-dir runs/final/unet_pretraining_800"),
        md("## 3b. Freezing ablation (Experiment 1)\n\n"
           "Four modes. `random_encoder` is the control that makes this an\n"
           "adjudication: a randomly re-initialized, frozen encoder. If it matches\n"
           "the frozen *pretrained* encoder, what transfers is not the features."),
        code("!python scripts/run_freezing_ablation.py \\\n"
             "    --config configs/config_final.yaml \\\n"
             "    --data-dir /content/eval \\\n"
             "    --run-dir runs/final/freezing \\\n"
             "    --modes encoder,decoder,random_encoder,none"),
        md("## 3c. Tune the INR learning rates in-distribution (recommended, ~20 min)\n\n"
           "The rates currently in the config were selected on `data/casestudy/`,\n"
           "which holds deliberately unusual out-of-distribution shapes. A partial\n"
           "in-distribution re-measurement showed the difference is real: IG-FWI\n"
           "scored 0.408x trivial on the case-study sample but 0.868x on eval case\n"
           "15002, and a grid rate of 3e-1 — merely mediocre there — **diverged**\n"
           "in distribution (6.5x). Run this, then paste the printed values into\n"
           "`configs/config_final.yaml` before the experiments below.\n\n"
           "Skip it only if you accept selecting hyperparameters on a different\n"
           "distribution from the one you report on, and say so in the thesis."),
        code("# --apply writes the winning values straight into the config, in place,\n"
             "# preserving comments and keeping a .bak. A rate that fails to beat the\n"
             "# trivial solution is NOT written -- the previous value stays, because\n"
             "# installing a rate that reconstructs nothing is worse than keeping the old one.\n"
             "!python scripts/tune_inr_learning_rates.py \\\n"
             "    --config configs/config_final.yaml \\\n"
             "    --data-dir /content/eval \\\n"
             "    --run-dir runs/final/inr_tuning \\\n"
             "    --epochs 5 \\\n"
             "    --apply"),
        md("Check what actually landed, then save the config to Drive. A Colab\n"
           "checkout is **not persistent** — without this the next session silently\n"
           "reverts to the untuned values."),
        code("import yaml\n"
             "c = yaml.safe_load(open('configs/config_final.yaml'))['experiments']\n"
             "for k in sorted(x for x in c if x.startswith('inr')):\n"
             "    v = c[k]\n"
             "    extra = ' '.join(f'{n}={v[n]}' for n in ('lr_grid', 'lr_bias') if n in v)\n"
             "    print(f\"{k:26s} lr={v['lr']:<10g} {extra}\")\n"
             "\n"
             "!mkdir -p {OUT}/session1\n"
             "!cp configs/config_final.yaml {OUT}/session1/config_final_tuned.yaml\n"
             "!cp runs/final/inr_tuning/tuned_values.yaml {OUT}/session1/ 2>/dev/null\n"
             "print('tuned config saved to Drive')"),
        md("## 3d. INR experiments (Experiment 2)\n\n"
           "Four ansätze on two held-out cases. Add the `_centered` variants to\n"
           "`--methods` if you want them; they roughly double the runtime."),
        code("!python scripts/run_all_experiments.py \\\n"
             "    --config configs/config_final.yaml \\\n"
             "    --methods inr_siren_fwi,inr_lr_fwi,inr_mpe_fwi,inr_ig_fwi \\\n"
             "    --cases 15000,15001 \\\n"
             "    --data-dir /content/eval \\\n"
             "    --run-dir runs/final/inr"),
        md("_Optional: the centred variants._"),
        code("# !python scripts/run_all_experiments.py \\\n"
             "#     --config configs/config_final.yaml \\\n"
             "#     --methods inr_siren_centered_fwi,inr_mpe_centered_fwi,inr_ig_centered_fwi \\\n"
             "#     --cases 15000,15001 --data-dir /content/eval \\\n"
             "#     --run-dir runs/final/inr_centered"),
    ]
    cells += save_cells("session1", "runs/final")
    return cells


def session2():
    cells = preamble(
        "Session 2 — Comparative pretraining (the long one)",
        "Trains every model family at every pretraining-set size, all into ONE run\n"
        "directory so the scaling curves are plotted in place. This is the longest\n"
        "session by a wide margin; keep the tab alive.")
    cells += [
        md("## 3a. U-Net and SegFormer at every sample count\n\n"
           "Sample counts come from `comparative_pretraining.sample_counts` in the\n"
           "config (250/500/1000/5000/10000/15000). Both families share seeds and\n"
           "sample ids at every size, so the comparison is paired."),
        code("!python scripts/temp_pretraining.py \\\n"
             "    --config configs/config_final.yaml \\\n"
             "    --data-dir /content/extended \\\n"
             "    --output-dir models/final \\\n"
             "    --run-dir runs/final/comparative_pretraining \\\n"
             "    --models unet,segformer"),
        md("## 3b. SegFormer with ImageNet initialization\n\n"
           "Only at 15000. This answers a different question from the scaling curve —\n"
           "whether out-of-domain pretraining substitutes for in-domain data — so it\n"
           "is compared against the random-init SegFormer at the same size."),
        code("!python scripts/temp_pretraining.py \\\n"
             "    --config configs/config_final.yaml \\\n"
             "    --data-dir /content/extended \\\n"
             "    --output-dir models/final \\\n"
             "    --run-dir runs/final/imagenet_pretraining \\\n"
             "    --sample-counts 15000 \\\n"
             "    --models segformer_imagenet"),
        md("### Checkpoint inventory\n\nSession 3 needs all of these."),
        code("import pathlib\n"
             "for p in sorted(pathlib.Path('models/final').iterdir()):\n"
             "    print(f'{p.stat().st_size/1e6:8.1f} MB  {p.name}')"),
    ]
    cells += save_cells("session2", "runs/final")
    return cells


def session3():
    cells = preamble(
        "Session 3 — Downstream evaluation and attention maps",
        "Runs the real FWI evaluation: each pretrained network is used as the\n"
        "reparameterization ansatz and driven through the full inversion loop.\n"
        "**Requires session 2's checkpoints.**",
        needs_models=True)
    cells += [
        md("## 3a. U-Net vs SegFormer, downstream FWI\n\n"
           "Every checkpoint inverted on 6 held-out cases. Note the eval cases have\n"
           "very different void fractions (0.18%–4.14%), so per-case numbers are\n"
           "written to CSV, not just the mean."),
        code("!python scripts/compare_unet_transformer.py \\\n"
             "    --config configs/config_final.yaml \\\n"
             "    --data-dir /content/eval \\\n"
             "    --model-dir models/final \\\n"
             "    --cases 15000,15001,15002,15003,15004,15005 \\\n"
             "    --sample-counts 250,500,1000,5000,10000,15000 \\\n"
             "    --models unet,segformer \\\n"
             "    --run-dir runs/final/compare_unet_segformer"),
        md("## 3b. ImageNet vs random initialization\n\n"
           "Same architecture, same data, same recipe — only the initial encoder\n"
           "weights differ."),
        code("!python scripts/compare_unet_transformer.py \\\n"
             "    --config configs/config_final.yaml \\\n"
             "    --data-dir /content/eval \\\n"
             "    --model-dir models/final \\\n"
             "    --cases 15000,15001,15002,15003,15004,15005 \\\n"
             "    --sample-counts 15000 \\\n"
             "    --models segformer,segformer_imagenet \\\n"
             "    --run-dir runs/final/compare_imagenet"),
        md("## 3c. Attention maps\n\n"
           "Caveat for the write-up: SegFormer reduces keys by the SR ratio, so each\n"
           "block attends to only ~32 key locations. These are NOT dense per-pixel\n"
           "ViT maps. What they show is where each output looks and how broadly."),
        code("!python scripts/visualize_segformer_attention.py \\\n"
             "    --config configs/config_final.yaml \\\n"
             "    --data-dir /content/eval \\\n"
             "    --checkpoint models/final/model_SegFormer_100_segmentation_15000.pt \\\n"
             "    --compare-checkpoint models/final/model_SegFormerImageNet_100_segmentation_15000.pt \\\n"
             "    --labels 'SegFormer (random init),SegFormer (ImageNet init)' \\\n"
             "    --case 15000 \\\n"
             "    --output-dir runs/final/attention"),
    ]
    cells += save_cells("session3", "runs/final")
    return cells


NOTEBOOKS = {
    "colab_session1_inr_freezing.ipynb": session1,
    "colab_session2_pretraining.ipynb": session2,
    "colab_session3_evaluation.ipynb": session3,
}

META = {
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
}


def main():
    for name, builder in NOTEBOOKS.items():
        nb = {"cells": builder(), "metadata": META,
              "nbformat": 4, "nbformat_minor": 0}
        path = REPO / name
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"  wrote {name}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
