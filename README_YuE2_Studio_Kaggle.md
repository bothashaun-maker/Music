# JesusJuiceProductions — YuE2 Studio 5.0.0 on Kaggle

Use the **actual upstream YuE2 Studio** notebooks in this branch:

1. `YuE2_Studio_Builder.ipynb` — builds the Linux Rust music service and pinned YuE2 CUDA/Vulkan engine from upstream tag `v5.0.0`, plus a verified Cloudflare binary.
2. `YuE2_Studio_Cloudflare.ipynb` — launches the upstream browser UI and service in Kaggle, enables the Studio's authenticated network access, and exposes the UI through Cloudflare Quick Tunnel.

## Run order

- Run the builder on Kaggle with Internet ON and Accelerator None.
- Save the notebook version with outputs.
- Attach those outputs to the runtime notebook as a Kaggle input.
- Run the runtime with Internet ON and GPU T4 ×2.
- Open the printed Cloudflare URL; keep the cell running.

The older `YuE2_Cover_Assets_Builder.ipynb` and `YuE2_ComfyUI_Cover_Studio.ipynb` are the previous ComfyUI workflow and are **not** the YuE2 Studio application.

## Scope and caveats

This is the upstream Studio 5.0.0 web interface running headlessly with the upstream Rust service and YuE2 engine. It is not the native Windows Tauri installer. Some desktop-shell-specific features may be unavailable in a browser/Kaggle session, and optional engines/components must be installed separately where the upstream Linux build supports them.

Model weights and optional engines have their own licences. The Studio's MIT licence does not change those terms.