import argparse
import inspect
from pathlib import Path

import torch
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from PIL import Image, ImageOps


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, deformed body, distorted hands, extra limbs, missing limbs, "
    "bad anatomy, duplicated person, inconsistent garment, wrong clothing, text, watermark, logo"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image multi-view inference (full model, caption prompt)")
    parser.add_argument("--garment_front", required=True, help="Path to the front-view garment image.")
    parser.add_argument("--garment_back", required=True, help="Path to the back-view garment image.")
    parser.add_argument(
        "--model_path",
        default="/home/work/MMSearch/cwf/search/model/FLUX.1-Fill-dev",
        help="Base FLUX Fill model path.",
    )
    parser.add_argument(
        "--transformer_path",
        required=True,
        help="Path to the trained transformer directory (or parent dir containing a 'transformer' subfolder).",
    )
    parser.add_argument("--output", default="output.png", help="Path for the generated output image.")
    parser.add_argument("--condition_output", default=None, help="Optional path to save the condition canvas.")
    parser.add_argument("--mask_output", default=None, help="Optional path to save the generation mask.")
    parser.add_argument("--cloth_type", default="garment", help="Garment category used in the prompt.")
    parser.add_argument("--gender", default="female", help="Model gender/style token used in the prompt.")
    parser.add_argument("--other_clothing", default="none", help="Optional paired clothing, e.g. jeans, skirt.")
    parser.add_argument("--ref_framing", default="same", help="Optional reference framing note.")
    parser.add_argument("--ref_extra_layers", default="none", help="Optional extra layers visible in references.")
    parser.add_argument("--prompt", default=None, help="Override the automatically built prompt.")
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


def build_prompt(args):
    if args.prompt:
        return args.prompt

    cloth = args.cloth_type
    gender = args.gender

    prompt = (
        f"a 7-grid layout: top row shows {cloth} front, {cloth} back, and a gray placeholder; "
        f"bottom row shows a {gender} model wearing the {cloth}"
    )

    if args.other_clothing and args.other_clothing.lower() != "none":
        prompt += f" with {args.other_clothing}"

    prompt += " in four views: reference"

    ref_details = []
    if args.ref_framing and args.ref_framing != "same":
        ref_details.append(f"{args.ref_framing} framing")
    if args.ref_extra_layers and args.ref_extra_layers.lower() != "none":
        ref_details.append(f"wearing additional {args.ref_extra_layers}")
    if ref_details:
        prompt += f" ({', '.join(ref_details)})"

    prompt += ", front, side, and back"
    return prompt


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


def resolve_transformer_path(path):
    p = Path(path)
    subfolder = p / "transformer"
    if subfolder.is_dir():
        return str(p), "transformer"
    return str(p), None


def load_pipeline(args):
    dtype = get_dtype(args.dtype)

    transformer_source, subfolder = resolve_transformer_path(args.transformer_path)
    load_kwargs = {"torch_dtype": dtype}
    if subfolder:
        load_kwargs["subfolder"] = subfolder
    transformer = FluxTransformer2DModel.from_pretrained(transformer_source, **load_kwargs)

    pipe = FluxFillPipeline.from_pretrained(
        args.model_path, transformer=transformer, torch_dtype=dtype,
    )
    if args.device_map:
        pipe.to(device_map=args.device_map)
    else:
        pipe.to(args.device)

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

    print("Loading pipeline (full model)...")
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
