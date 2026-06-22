"""DualViewFashion image inference.

Generate synchronized multi-view fashion model images from dual-view garment
references. The script builds a masked FLUX Fill canvas with garment reference
views on the top row and asks the model to generate model views on the bottom
row in a single forward pass.
"""

import argparse
import inspect
from pathlib import Path

import torch
from diffusers import FluxFillPipeline
from PIL import Image, ImageOps


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, deformed body, distorted hands, extra limbs, missing limbs, "
    "bad anatomy, duplicated person, inconsistent garment, wrong clothing, text, watermark, logo"
)


def parse_args():
    parser = argparse.ArgumentParser(description="DualViewFashion image inference")
    parser.add_argument("--garment_front", required=True, help="Path to the front-view garment reference image.")
    parser.add_argument("--garment_back", required=True, help="Path to the back-view garment reference image.")
    parser.add_argument(
        "--model_path",
        default="black-forest-labs/FLUX.1-Fill-dev",
        help="Base FLUX Fill model path or Hugging Face repo id.",
    )
    parser.add_argument("--lora_path", required=True, help="Path to the DualViewFashion LoRA checkpoint.")
    parser.add_argument("--output", default="dualviewfashion_output.png", help="Path for the generated output grid.")
    parser.add_argument("--condition_output", default=None, help="Optional path for the masked condition canvas.")
    parser.add_argument("--mask_output", default=None, help="Optional path for the generation mask.")
    parser.add_argument("--prompt", default=None, help="Override the default generation prompt.")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--cloth_size", type=int, default=512, help="Reference slot size. 512 creates a 1536x1024 grid.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device_map",
        default=None,
        help="Optional diffusers device_map. If unset, the pipeline is moved to --device.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=("bf16", "fp16", "fp32"),
        help="Torch dtype for inference.",
    )
    return parser.parse_args()


def get_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def load_rgb(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


PROMPT = "a 7-grid layout: top row shows clothing front, clothing back, and a gray placeholder; bottom row shows a model wearing the clothing in four views: reference, front, side, and back"


def build_prompt(args):
    if args.prompt:
        return args.prompt
    return PROMPT


def build_grid_and_mask(front_path, back_path, cloth_size):
    grid_w = 3 * cloth_size
    grid_h = 2 * cloth_size

    front = load_rgb(front_path).resize((cloth_size, cloth_size), Image.Resampling.LANCZOS)
    back = load_rgb(back_path).resize((cloth_size, cloth_size), Image.Resampling.LANCZOS)
    gray = Image.new("RGB", (cloth_size, cloth_size), (128, 128, 128))

    grid = Image.new("RGB", (grid_w, grid_h), (128, 128, 128))
    grid.paste(front, (0, 0))
    grid.paste(back, (cloth_size, 0))
    grid.paste(gray, (2 * cloth_size, 0))

    mask = Image.new("L", (grid_w, grid_h), 0)
    mask.paste(Image.new("L", (grid_w, cloth_size), 255), (0, cloth_size))
    return grid, mask


def load_pipeline(args):
    dtype = get_dtype(args.dtype)
    load_kwargs = {"torch_dtype": dtype}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map

    pipe = FluxFillPipeline.from_pretrained(args.model_path, **load_kwargs)
    if not args.device_map:
        pipe.to(args.device)

    pipe.load_lora_weights(args.lora_path)
    return pipe


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(args)
    grid, mask = build_grid_and_mask(args.garment_front, args.garment_back, args.cloth_size)

    if args.condition_output:
        Path(args.condition_output).parent.mkdir(parents=True, exist_ok=True)
        grid.save(args.condition_output)
    if args.mask_output:
        Path(args.mask_output).parent.mkdir(parents=True, exist_ok=True)
        mask.save(args.mask_output)

    print("Loading DualViewFashion image pipeline...")
    pipe = load_pipeline(args)
    print(f"Prompt: {prompt}")

    generator_device = args.device if args.device.startswith("cuda") else "cpu"
    generator = torch.Generator(generator_device).manual_seed(args.seed)

    call_kwargs = dict(
        prompt=prompt,
        image=grid,
        mask_image=mask,
        height=grid.height,
        width=grid.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        max_sequence_length=512,
        generator=generator,
    )
    if "negative_prompt" in inspect.signature(pipe.__call__).parameters:
        call_kwargs["negative_prompt"] = args.negative_prompt

    result = pipe(**call_kwargs).images[0]

    result.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
