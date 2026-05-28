import gradio as gr
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.registry import EXPERIMENTS
from src.ui import callbacks
from src.ui.case_viewer import case_choices, view_case
from src.ui.config_forms import default_form_values, experiment_form_values
from src.config import load_config


def build_app():
    methods = list(EXPERIMENTS.keys())
    cases = case_choices()
    defaults = default_form_values(method_name=methods[0])
    config_defaults = load_config("configs/default.yaml")
    pre_cfg = config_defaults["pretraining"]
    unet_cfg = config_defaults.get("models", {}).get("unet", {})

    with gr.Blocks(title="FWI Experiments, Data Generation and Pretraining UI") as app:
        gr.Markdown("# FWI Experiments, Data Generation and Pretraining UI")

        with gr.Tab("Run Experiments"):
            with gr.Row():
                method = gr.Dropdown(methods, value=methods[0], label="Method")
                selected_cases = gr.Dropdown(cases, multiselect=True, label="Cases (Materials)")
                refresh = gr.Button("Refresh Cases")

            config_mode = gr.Radio(
                ["Use existing YAML path", "Create custom YAML from UI parameters"],
                value="Use existing YAML path",
                label="Config Mode",
            )
            yaml_path = gr.Textbox("configs/default.yaml", label="YAML Path")

            with gr.Group(visible=False) as custom_group:
                custom_config_name = gr.Textbox("ui_custom.yaml", label="Custom Config Name")
                gr.Markdown("Shared parameters")
                with gr.Row():
                    Lx = gr.Number(defaults["Lx"], label="Lx")
                    Ly = gr.Number(defaults["Ly"], label="Ly")
                    Nx = gr.Number(defaults["Nx"], label="Nx", precision=0)
                    Ny = gr.Number(defaults["Ny"], label="Ny", precision=0)
                with gr.Row():
                    dt = gr.Number(defaults["dt"], label="dt")
                    N = gr.Number(defaults["N"], label="N", precision=0)
                    gamma0 = gr.Number(defaults["gamma0"], label="gamma0")
                with gr.Row():
                    rho = gr.Number(defaults["rho"], label="rho")
                    c = gr.Number(defaults["c"], label="c")
                    frequency = gr.Number(defaults["frequency"], label="frequency")
                with gr.Row():
                    cycles = gr.Number(defaults["cycles"], label="cycles", precision=0)
                    amplitude = gr.Number(defaults["amplitude"], label="amplitude")
                    number_of_sources = gr.Number(defaults["number_of_sources"], label="number_of_sources", precision=0)
                with gr.Row():
                    distance_between_sources = gr.Number(defaults["distance_between_sources"], label="distance_between_sources", precision=0)
                    distance_between_sensors = gr.Number(defaults["distance_between_sensors"], label="distance_between_sensors", precision=0)

                gr.Markdown("Experiment parameters")
                with gr.Row():
                    epochs = gr.Number(defaults["epochs"], label="epochs", precision=0)
                    learning_rate = gr.Number(defaults["learning_rate"], label="learning_rate")
                    clip_grad = gr.Number(defaults["clip_grad"], label="clip_grad")
                    cost_scaling = gr.Number(defaults["cost_scaling"], label="cost_scaling")
                with gr.Row():
                    l2 = gr.Number(defaults["l2"], label="l2")
                    alpha = gr.Number(defaults["alpha"], label="alpha")
                    beta = gr.Number(defaults["beta"], label="beta")
                pretrained_model_path = gr.Textbox(defaults["pretrained_model_path"], label="Pretrained Model Directory")

            run = gr.Button("Run Selected Cases", variant="primary")
            status = gr.Textbox(label="Status", lines=8)
            elapsed = gr.Textbox(label="Total Elapsed Time")
            run_dirs = gr.Textbox(label="Run Directories", lines=4)
            console = gr.Textbox(label="Python Output", lines=12)
            gallery = gr.Gallery(label="Outputs", columns=2, height=520, preview=True)

            def mode_change(mode):
                state = callbacks.toggle_config_mode(mode)
                return gr.update(visible=state["yaml_visible"]), gr.update(visible=state["custom_visible"])

            def method_change(method_name):
                values = experiment_form_values(method_name=method_name)
                return (
                    values["epochs"],
                    values["learning_rate"],
                    values["clip_grad"],
                    values["cost_scaling"],
                    values["l2"],
                    values["alpha"],
                    values["beta"],
                    values["pretrained_model_path"],
                )

            method.change(
                method_change,
                inputs=method,
                outputs=[
                    epochs,
                    learning_rate,
                    clip_grad,
                    cost_scaling,
                    l2,
                    alpha,
                    beta,
                    pretrained_model_path,
                ],
            )
            config_mode.change(mode_change, inputs=config_mode, outputs=[yaml_path, custom_group])
            refresh.click(lambda: gr.update(choices=case_choices()), outputs=selected_cases)
            run.click(
                callbacks.run_experiments_from_ui,
                inputs=[
                    method,
                    config_mode,
                    yaml_path,
                    custom_config_name,
                    selected_cases,
                    Lx,
                    Ly,
                    Nx,
                    Ny,
                    dt,
                    N,
                    gamma0,
                    rho,
                    c,
                    frequency,
                    cycles,
                    amplitude,
                    number_of_sources,
                    distance_between_sources,
                    distance_between_sensors,
                    epochs,
                    learning_rate,
                    clip_grad,
                    cost_scaling,
                    l2,
                    alpha,
                    beta,
                    pretrained_model_path,
                ],
                outputs=[status, elapsed, run_dirs, gallery, console],
            )

        with gr.Tab("Data Generator / Viewer"):
            case_select = gr.Dropdown(cases, label="Case")
            refresh_viewer = gr.Button("Refresh Cases")
            material_image = gr.Image(label="Material Field")
            case_info = gr.Textbox(label="Case Info", lines=6)
            case_select.change(view_case, inputs=case_select, outputs=[material_image, case_info])
            refresh_viewer.click(lambda: gr.update(choices=case_choices()), outputs=case_select)

            gr.Markdown("## Train/Test Split")
            split_mode = gr.Radio(
                ["Use existing train/test folders", "Split data/raw into train/test"],
                value="Use existing train/test folders",
                label="Dataset Split Mode",
            )
            with gr.Row():
                raw_data_dir = gr.Textbox("data/raw", label="Raw Data Directory")
                split_train_dir = gr.Textbox("data/train", label="Train Output Directory")
                split_test_dir = gr.Textbox("data/test", label="Test Output Directory")
            with gr.Row():
                train_ratio = gr.Slider(0.05, 0.95, value=0.8, step=0.05, label="Train Ratio")
                split_seed = gr.Number(2, label="Split Seed", precision=0)
                split_overwrite = gr.Checkbox(False, label="Overwrite existing split files")
            split_raw = gr.Button("Create Train/Test Split")
            split_status = gr.Textbox(label="Split Status", lines=8)
            split_raw.click(
                callbacks.split_raw_dataset_from_ui,
                inputs=[
                    split_mode,
                    raw_data_dir,
                    split_train_dir,
                    split_test_dir,
                    train_ratio,
                    split_seed,
                    split_overwrite,
                ],
                outputs=[split_status, case_select, selected_cases],
            )

            gr.Markdown("## Create Custom Case")
            canvas_config_path = gr.Textbox("configs/default.yaml", label="Config Path")
            canvas_output_dir = gr.Textbox("data/custom", label="Output Directory")
            canvas_case_name = gr.Textbox("0", label="Custom Case Name or ID")
            generate_measurement_gradient = gr.Checkbox(True, label="Generate measurement and gradient")
            with gr.Row():
                init_canvas = gr.Button("Initialize Canvas")
                preview_gamma = gr.Button("Preview Gamma")
                generate_custom = gr.Button("Generate Custom Case", variant="primary")
            drawing_canvas = gr.ImageEditor(label="Draw damage/void regions. Select blue color for voids only.", type="numpy")
            gamma_preview = gr.Image(label="Gamma Preview")
            canvas_status = gr.Textbox(label="Canvas Status", lines=8)

            init_canvas.click(
                callbacks.initialize_canvas,
                inputs=canvas_config_path,
                outputs=[drawing_canvas, canvas_status],
            )
            preview_gamma.click(
                callbacks.preview_canvas_gamma,
                inputs=[drawing_canvas, canvas_config_path],
                outputs=[gamma_preview, canvas_status],
            )
            generate_custom.click(
                callbacks.generate_custom_case_from_canvas,
                inputs=[
                    drawing_canvas,
                    canvas_case_name,
                    canvas_output_dir,
                    canvas_config_path,
                    generate_measurement_gradient,
                ],
                outputs=[gamma_preview, canvas_status, case_select],
            )

        with gr.Tab("Pretrain Models"):
            pretrain_config_path = gr.Textbox("configs/default.yaml", label="YAML Config Path")
            pretrain_data_dir = gr.Textbox("data/train", label="Training Data Directory")
            pretrain_output_dir = gr.Textbox("models/pretrained", label="Output Model Directory")
            pretrain_model_name = gr.Textbox(pre_cfg.get("model_type", "Unet"), label="Model Name")
            with gr.Row():
                pretrain_samples = gr.Number(pre_cfg["numberOfSamples"], label="number_of_samples", precision=0)
                pretrain_epochs = gr.Number(pre_cfg["epochs"], label="epochs", precision=0)
                pretrain_batch_size = gr.Number(
                    max(1, int(pre_cfg["numberOfSamples"]) // int(pre_cfg["batchDivisor"])),
                    label="batch_size",
                    precision=0,
                )
            with gr.Row():
                pretrain_lr = gr.Number(pre_cfg["lr"], label="learning_rate")
                pretrain_clip_grad = gr.Number(pre_cfg["clipGrad"], label="clip_grad")
                pretrain_l2 = gr.Number(pre_cfg["l2"], label="l2")
                pretrain_alpha = gr.Number(pre_cfg["alpha"], label="alpha")
                pretrain_beta = gr.Number(pre_cfg["beta"], label="beta")
            pretrain_channels = gr.Textbox(
                ",".join(str(item) for item in unet_cfg.get("channels", pre_cfg["NNchannels"])),
                label="U-Net Channels",
            )
            pretrain_conv_count = gr.Number(
                unet_cfg.get("number_of_convolutions_per_block", pre_cfg["numberOfConvolutionsPerBlock"]),
                label="number_of_convolutions_per_block",
                precision=0,
            )
            pretrain_batch_norm = gr.Checkbox(unet_cfg.get("batch_norm", True), label="batch_norm")
            start_pretrain = gr.Button("Start Pretraining", variant="primary")
            pretrain_status = gr.Textbox(label="Pretraining Status", lines=10)
            pretrain_model_path = gr.Textbox(label="Saved Model Path")
            pretrain_elapsed = gr.Textbox(label="Elapsed Time")
            pretrain_loss_plot = gr.Image(label="Training / Validation Loss Plot")

            start_pretrain.click(
                callbacks.start_pretraining_from_ui,
                inputs=[
                    pretrain_config_path,
                    pretrain_data_dir,
                    pretrain_output_dir,
                    pretrain_model_name,
                    pretrain_samples,
                    pretrain_epochs,
                    pretrain_batch_size,
                    pretrain_lr,
                    pretrain_clip_grad,
                    pretrain_l2,
                    pretrain_alpha,
                    pretrain_beta,
                    pretrain_channels,
                    pretrain_conv_count,
                    pretrain_batch_norm,
                ],
                outputs=[pretrain_status, pretrain_model_path, pretrain_elapsed, pretrain_loss_plot],
            )

    return app


if __name__ == "__main__":
    build_app().launch()
