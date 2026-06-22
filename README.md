# DualViewFashion

### Dual-View Garment-Conditioned Multi-View Fashion Model Generation

> A FLUX-Fill based generative framework that synthesizes four coherent fashion-model views from dual-view garment references in a single inpainting forward pass.

---

DualViewFashion formulates multi-view fashion model generation as a **structured inpainting** problem. Given paired garment views, typically the front and back of a clothing item, we construct a canonical 7-region canvas and use **FLUX Fill** to complete the masked model-view regions. The system generates a synchronized set of model images, including reference-pose, front, side, and back views, without editing an existing person image.

This is **not** a virtual try-on or person-inpainting pipeline. DualViewFashion creates a new fashion model presentation conditioned by garment references.

## Highlights

- **Dual-View Garment Conditioning.** Instead of relying on a single product image, DualViewFashion conditions generation on paired garment views. The front view provides appearance cues such as color, print, neckline, and silhouette; the back view preserves reverse-side structure and rear-view design details.

- **Single-Pass Four-View Synthesis.** We cast multi-view generation into a single FLUX-Fill inpainting canvas. One forward pass produces four model views: reference-pose, front, side, and back. This avoids independently sampling each viewpoint and reduces cross-view drift.

- **Layout-Constrained Inpainting.** The model receives a 7-region layout: two garment reference slots, one identity placeholder slot, and four masked model-view slots. The inpainting objective encourages the model to respect the garment evidence while completing all target views jointly.

- **Identity Placeholder for Future Control.** The gray slot in the reference row is intentionally reserved. In the current release it is filled with a neutral gray placeholder and annotated in the examples; future versions will replace this slot with a model identity reference, enabling generation of a fixed model identity wearing the target garment.

- **Image-First, Video-Ready.** The core contribution is multi-view image generation. We also provide an optional Wan2.2 multi-frame reference video inference script, where multiple reference images are injected as sparse temporal conditions.

## TODO

- [x] Image inference code release
- [x] Wan2.2 multi-frame reference video inference code release
- [x] Example gallery release
- [x] Low-resolution full-model checkpoint release (512 x 384)
- [x] High-resolution full-model checkpoint ready (1024 x 768)
- [x] Image training code release
- [x] Wan2.2 video checkpoint release
- [ ] Wan2.2 video training code release
- [ ] Dataset release
- [ ] Model-identity conditioned generation

## Examples

The examples below are selected from the released full-parameter image checkpoints. The output canvas contains the dual-view garment references in the first row and four generated model views in the second row. The top-right gray region is labeled `Future Model ID` to match the reserved model-identity placeholder style.

| Example | Generated multi-view canvas |
|:---:|:---:|
| Example 1<br>(512 full model) | <img src="assets/examples/example_01.jpg" width="420"> |
| Example 2<br>(512 full model) | <img src="assets/examples/example_02.jpg" width="420"> |
| Example 3<br>(1024 full model, step 1000, actively improving) | <img src="assets/examples/example_03.jpg" width="640"> |


### Video Example

The optional video extension uses multiple reference frames as sparse temporal conditions. The example below shows three reference frames placed at relative positions `0.25`, `0.5`, and `0.75`.

| Reference Frames | Generated Video |
|:---:|:---:|
| <img src="assets/video/ref_0250.jpg" width="96"> <img src="assets/video/ref_0500.jpg" width="96"> <img src="assets/video/ref_0750.jpg" width="96"> | <img src="assets/video/dualviewfashion_video_preview.gif" width="300"><br>[Download MP4](assets/video/dualviewfashion_video_example.mp4) |

The current video demo is a 10-second, 161-frame generation. We plan to release a 10-second, 241-frame version in a future checkpoint release. The README uses a compact animated preview for reliable inline playback; the original MP4 is kept as a downloadable asset.

## Quick Start

### Environment

```bash
conda create -n search-train python=3.10 -y
conda activate search-train
pip install -r requirements.txt
```

DualViewFashion image inference was tested with `diffusers` `0.39.0.dev0` and a `FluxFillPipeline`-compatible FLUX Fill implementation.

### Image Inference

```bash
python inference_full_caption.py \
  --garment_front path/to/garment_front.jpg \
  --garment_back path/to/garment_back.jpg \
  --model_path path/to/FLUX.1-Fill-dev \
  --transformer_path path/to/trained-full-multiview-caption-stage1 \
  --output outputs/dualviewfashion_grid.png \
  --cloth_type "dress" \
  --gender "female" \
  --cloth_size 512 \
  --steps 50 \
  --guidance_scale 30.0 \
  --seed 42
```

The default command uses the 512 x 384 full-model checkpoint. For the 1024 x 768 full-model checkpoint, use `--transformer_path path/to/trained-full-multiview-caption-stage2/checkpoint-1000` and `--cloth_size 1024`. The 1024 version is initialized from the 512 version and trained for another 1000 steps.

The generated image is a single 7-region canvas. The bottom row contains four synchronized model renderings produced by one inpainting pass.

### Optional Video Inference

```bash
python video_inference.py \
  --model_dir path/to/Wan2.2-I2V-A14B \
  --ref_images ref_000.png ref_025.png ref_050.png ref_075.png \
  --frame_positions 0 0.25 0.5 0.75 \
  --high_lora path/to/dualviewfashion-wan22-high-noise.safetensors \
  --low_lora path/to/dualviewfashion-wan22-low-noise.safetensors \
  --prompt "a fashion model walks in, poses, turns around, and walks out, full body, studio lighting" \
  --output outputs/dualviewfashion_video.mp4
```

The video LoRA checkpoints are available at [ShineChen1024/DualViewFashion](https://huggingface.co/ShineChen1024/DualViewFashion). Current video inference examples use 161 frames for 10-second generation; a 241-frame, 10-second version is planned.

## Checkpoints

Checkpoints are hosted on Hugging Face: [ShineChen1024/DualViewFashion](https://huggingface.co/ShineChen1024/DualViewFashion)

| Model | Path | Status |
|---|---|---|
| DualViewFashion full model (512 x 384) | `trained-full-multiview-caption-stage1/transformer` | [Available](https://huggingface.co/ShineChen1024/DualViewFashion) |
| DualViewFashion full model (1024 x 768) | `trained-full-multiview-caption-stage2/checkpoint-1000/transformer` | [Available](https://huggingface.co/ShineChen1024/DualViewFashion) |
| DualViewFashion Wan2.2 video LoRA | - | [Available](https://huggingface.co/ShineChen1024/DualViewFashion) |
| DualViewFashion dataset | - | Coming soon |

## Architecture

```text
                 +------------------------------------------------+
                 |                DualViewFashion                 |
                 |          FLUX-Fill Layout Inpainting           |
                 |                                                |
Garment Front -->|  [ Garment Front ] [ Garment Back ] [ ID Slot ]|
Garment Back  -->|                                                |
Identity Slot -->|  [ Ref-Pose ] [ Front ] [ Side ] [ Back Model ]|--> Four model views
Masked Targets ->|                                                |
                 +------------------------------------------------+
```

**Key difference from virtual try-on pipelines:**

| | Virtual Try-On / Person Inpainting | DualViewFashion |
|---|---|---|
| Input person | Required | Not required |
| Target task | Edit clothing on an existing person | Generate new fashion model views |
| Garment evidence | Often single-view | Dual-view garment references |
| View generation | Usually one edited image | Four coordinated model views |
| Generation process | Person-region inpainting | Layout-conditioned multi-view inpainting |

## How It Works

1. **Canvas Construction.** The front and back garment images are placed in the first two reference slots. A neutral gray identity placeholder is placed in the third reference slot.
2. **Mask Definition.** The complete bottom row is masked as the target region for generation.
3. **FLUX-Fill Inpainting.** A LoRA-adapted FLUX Fill model completes the masked region in one pass, producing four synchronized fashion-model views.
4. **Future Identity Control.** The gray placeholder will be replaced by a model identity reference, allowing the same model identity to be rendered wearing different garments.

## Training

DualViewFashion adopts a two-stage full-parameter training curriculum for the image model:

- **Stage 1 - Low-Resolution Training (512 x 384).** The model first learns the canonical 7-region inpainting formulation, dual-view garment conditioning, and coarse cross-view correspondence at 512 x 384 resolution.
- **Stage 2 - High-Resolution Training (1024 x 768).** Starting from the 512 x 384 full-model checkpoint, we continue full-parameter training at 1024 x 768 resolution for another 1000 steps to improve garment texture fidelity, silhouette quality, view consistency, and back-view detail preservation.

Both released image checkpoints are full FLUX transformer checkpoints, not LoRA adapters. Training code is available under the `train/` directory. Dataset release is coming soon.

Future extensions include model-identity conditioning through the reserved identity placeholder and Wan2.2 multi-frame reference video training.

## Project Structure

```text
DualViewFashion/
|-- inference.py                  # FLUX-Fill image inference (fixed prompt)
|-- inference_caption.py          # FLUX-Fill image inference (caption prompt, LoRA)
|-- inference_full_caption.py     # FLUX-Fill image inference (caption prompt, full model)
|-- video_inference.py            # Wan2.2 multi-frame reference video inference
|-- train/
|   |-- finetune-mm-stage1.sh                        # LoRA training stage 1 (fixed prompt)
|   |-- finetune-mm-stage2.sh                        # LoRA training stage 2 (fixed prompt)
|   |-- finetune-mm-caption-stage1.sh                # LoRA training stage 1 (caption prompt)
|   |-- finetune-mm-caption-stage2.sh                # LoRA training stage 2 (caption prompt)
|   |-- finetune-mm-caption-stage1-full.sh           # Full training stage 1 (caption prompt)
|   |-- finetune-mm-caption-stage2-full.sh           # Full training stage 2 (caption prompt)
|   |-- finetune-multiview.py                        # LoRA training script (fixed prompt)
|   |-- finetune-multiview-caption.py                # LoRA training script (caption prompt)
|   |-- finetune-multiview-caption-full-deepspeed.py # Full training script (caption prompt)
|   `-- ds_config_zero2.json                         # DeepSpeed ZeRO-2 config
|-- assets/examples/              # Generated examples
|-- requirements.txt
|-- README.md
`-- LICENSE
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{dualviewfashion2026,
    title={DualViewFashion: Dual-View Garment-Conditioned Multi-View Fashion Model Generation},
    author={},
    year={2026}
}
```

## Acknowledgments

This project builds upon FLUX Fill for layout-conditioned image inpainting and Wan2.2/DiffSynth for optional multi-frame reference video generation.
