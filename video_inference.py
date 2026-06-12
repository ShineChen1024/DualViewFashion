"""DualViewFashion Wan2.2 multi-frame reference video inference.

This optional extension generates a fashion video from multiple reference images
by injecting them as sparse frame conditions into Wan2.2 I2V. It is released as
inference-only code; training code and checkpoints are coming soon.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline, WanVideoUnit_ImageEmbedderVAE
from diffsynth.utils.data import save_video


DEFAULT_NEGATIVE_PROMPT = (
    "overexposed, static, blurry details, subtitles, text, watermark, low quality, worst quality, "
    "jpeg artifacts, ugly, deformed, bad anatomy, bad hands, bad face, extra fingers, fused fingers, "
    "extra limbs, missing limbs, duplicated body, cluttered background, multiple people, walking backwards"
)


def parse_args():
    parser = argparse.ArgumentParser(description="DualViewFashion Wan2.2 multi-frame reference video inference")
    parser.add_argument("--model_dir", required=True, help="Path to the base Wan2.2-I2V-A14B model directory.")
    parser.add_argument(
        "--high_lora",
        default=None,
        help="Path to the high-noise DualViewFashion video LoRA checkpoint. Coming soon.",
    )
    parser.add_argument(
        "--low_lora",
        default=None,
        help="Path to the low-noise DualViewFashion video LoRA checkpoint. Coming soon.",
    )
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA merge strength.")
    parser.add_argument(
        "--ref_images",
        nargs="+",
        required=True,
        help="Reference image paths in the same order as --frame_positions.",
    )
    parser.add_argument(
        "--frame_positions",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75],
        help="Relative frame positions for each reference image, from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--prompt",
        default="a fashion model walks in, poses, turns around, and walks out, full body, studio lighting",
        help="Text prompt for video generation.",
    )
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", default="outputs/dualviewfashion_video.mp4")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--num_frames", type=int, default=161)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--switch_dit_boundary", type=float, default=0.875)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--no_tiling", action="store_true", help="Disable VAE tiling.")
    parser.add_argument("--tile_size", type=int, nargs=2, default=(30, 52), metavar=("H", "W"))
    parser.add_argument("--tile_stride", type=int, nargs=2, default=(15, 26), metavar=("H", "W"))
    return parser.parse_args()


def torch_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def load_rgb(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def validate_args(args):
    if len(args.ref_images) != len(args.frame_positions):
        raise ValueError(
            f"Number of --ref_images ({len(args.ref_images)}) must match "
            f"--frame_positions ({len(args.frame_positions)})."
        )
    for position in args.frame_positions:
        if position < 0.0 or position > 1.0:
            raise ValueError(f"Frame position must be in [0, 1], got {position}.")


def build_frame_indices(frame_positions, num_frames):
    return [min(max(round((num_frames - 1) * pos), 0), num_frames - 1) for pos in frame_positions]


def build_multiframe_vae_embedding(
    pipe,
    reference_images,
    frame_positions,
    num_frames,
    height,
    width,
    tiled=True,
    tile_size=(30, 52),
    tile_stride=(15, 26),
):
    indices = build_frame_indices(frame_positions, num_frames)
    pipe.load_models_to_device(["vae"])

    vae_input = torch.zeros(
        3,
        num_frames,
        height,
        width,
        dtype=pipe.torch_dtype,
        device=pipe.device,
    )
    mask = torch.zeros(1, num_frames, height // 8, width // 8, device=pipe.device)

    for index, image in zip(indices, reference_images):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        image_tensor = pipe.preprocess_image(image).to(dtype=pipe.torch_dtype, device=pipe.device)
        vae_input[:, index:index + 1] = image_tensor.transpose(0, 1)
        mask[:, index:index + 1] = 1

    mask = torch.concat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)
    mask = mask.view(1, mask.shape[1] // 4, 4, height // 8, width // 8)
    mask = mask.transpose(1, 2)[0]

    y = pipe.vae.encode(
        [vae_input],
        device=pipe.device,
        tiled=tiled,
        tile_size=tuple(tile_size),
        tile_stride=tuple(tile_stride),
    )[0]
    y = torch.concat([mask, y.to(dtype=pipe.torch_dtype, device=pipe.device)])
    return y.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)


class MultiFrameImageEmbedderVAE(WanVideoUnit_ImageEmbedderVAE):
    """Wan image embedder that injects multiple sparse reference frames."""

    def __init__(self, reference_images, frame_positions):
        super().__init__()
        self.reference_images = reference_images
        self.frame_positions = frame_positions

    def process(self, pipe, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        return {
            "y": build_multiframe_vae_embedding(
                pipe,
                self.reference_images,
                self.frame_positions,
                num_frames,
                height,
                width,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        }


def build_wan22_model_configs(model_dir):
    model_dir = Path(model_dir)
    high_shards = [
        str(model_dir / "high_noise_model" / f"diffusion_pytorch_model-0000{i}-of-00006.safetensors")
        for i in range(1, 7)
    ]
    low_shards = [
        str(model_dir / "low_noise_model" / f"diffusion_pytorch_model-0000{i}-of-00006.safetensors")
        for i in range(1, 7)
    ]
    return [
        ModelConfig(path=high_shards),
        ModelConfig(path=low_shards),
        ModelConfig(path=str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth")),
        ModelConfig(path=str(model_dir / "Wan2.1_VAE.pth")),
    ], ModelConfig(path=str(model_dir / "google" / "umt5-xxl"))


def load_pipeline(args):
    model_configs, tokenizer_config = build_wan22_model_configs(args.model_dir)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype(args.dtype),
        device=args.device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    if args.high_lora:
        pipe.load_lora(pipe.dit, args.high_lora, alpha=args.lora_alpha)
        print(f"Loaded high-noise LoRA: {args.high_lora}")
    if args.low_lora:
        pipe.load_lora(pipe.dit2, args.low_lora, alpha=args.lora_alpha)
        print(f"Loaded low-noise LoRA: {args.low_lora}")
    if not args.high_lora and not args.low_lora:
        print("No video LoRA checkpoints provided; running the base Wan2.2 model with multi-frame conditioning.")
    return pipe


def replace_image_embedder(pipe, reference_images, frame_positions):
    for index, unit in enumerate(pipe.units):
        if isinstance(unit, WanVideoUnit_ImageEmbedderVAE):
            pipe.units[index] = MultiFrameImageEmbedderVAE(reference_images, frame_positions)
            print(f"Replaced Wan image embedder unit {index} with multi-frame reference conditioning.")
            return
    raise RuntimeError("WanVideoUnit_ImageEmbedderVAE was not found in the Wan pipeline units.")


def main():
    args = parse_args()
    validate_args(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    reference_images = [load_rgb(path) for path in args.ref_images]
    pipe = load_pipeline(args)
    replace_image_embedder(pipe, reference_images, args.frame_positions)

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=reference_images[0],
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        switch_DiT_boundary=args.switch_dit_boundary,
        tiled=not args.no_tiling,
        tile_size=tuple(args.tile_size),
        tile_stride=tuple(args.tile_stride),
    )

    save_video(video, args.output, fps=args.fps, quality=args.quality)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
