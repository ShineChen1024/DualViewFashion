import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
import json
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from diffusers.image_processor import VaeImageProcessor

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

check_min_version("0.36.0.dev0")

logger = get_logger(__name__)


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="FLUX Fill LoRA training with per-sample captions.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="flux-multiview-lora")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--weighting_scheme", type=str, default="none",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--prodigy_beta3", type=float, default=None)
    parser.add_argument("--prodigy_decouple", type=bool, default=True)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04)
    parser.add_argument("--adam_weight_decay_text_encoder", type=float, default=1e-03)
    parser.add_argument("--lora_layers", type=str, default=None)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--prodigy_use_bias_correction", type=bool, default=True)
    parser.add_argument("--prodigy_safeguard_warmup", type=bool, default=True)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--upcast_before_saving", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--cloth_size", type=int, default=512)
    parser.add_argument("--resume_lora_path", type=str, default=None,
                        help="Path to pre-trained LoRA weights for stage-2 fine-tuning")

    parser.add_argument("--jsonl_path", type=str,
                        default="/home/work/MMSearch/cwf/search/multi-view/all_two_front_accept_with_candidate_captioned.jsonl")
    parser.add_argument("--image_dir", type=str,
                        default="/home/work/MMSearch/cwf/search/multi-view/images")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


# Grid layout (sizes set from --cloth_size):
# Top row:    cloth_front | cloth_back | gray_ID
# Bottom row: reference   | front      | side    | back

CLOTH_SIZE = 512
MODEL_W = 384
MODEL_H = 512
GRID_W = 3 * CLOTH_SIZE
GRID_H = 2 * CLOTH_SIZE


def build_prompt(caption):
    cloth = caption.get("cloth_type", "clothing")
    gender = caption.get("gender", "female")
    other = caption.get("other_clothing", "none")
    ref_framing = caption.get("ref_framing", "same")
    ref_extra = caption.get("ref_extra_layers", "none")

    prompt = (
        f"a 7-grid layout: top row shows {cloth} front, {cloth} back, and a gray placeholder; "
        f"bottom row shows a {gender} model wearing the {cloth}"
    )

    if other and other.lower() != "none":
        prompt += f" with {other}"

    prompt += " in four views: reference"

    ref_details = []
    if ref_framing and ref_framing != "same":
        ref_details.append(f"{ref_framing} framing")
    if ref_extra and ref_extra.lower() != "none":
        ref_details.append(f"wearing additional {ref_extra}")
    if ref_details:
        prompt += f" ({', '.join(ref_details)})"

    prompt += ", front, side, and back"
    return prompt


class MultiViewDataset(Dataset):
    def __init__(self, jsonl_path, image_dir):
        self.image_dir = image_dir
        self.datas = [json.loads(line) for line in open(jsonl_path)]
        self.prompts = [build_prompt(d.get("caption", {})) for d in self.datas]
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.gray_id = Image.new("RGB", (CLOTH_SIZE, CLOTH_SIZE), (128, 128, 128))

    def __len__(self):
        return len(self.datas)

    def _load_cloth(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((CLOTH_SIZE, CLOTH_SIZE), Image.LANCZOS)
        return img

    def _load_model(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((CLOTH_SIZE, CLOTH_SIZE), Image.LANCZOS)
        left = (CLOTH_SIZE - MODEL_W) // 2
        img = img.crop((left, 0, left + MODEL_W, CLOTH_SIZE))
        return img

    def __getitem__(self, index):
        data = self.datas[index]
        pid = data["id"]
        base = os.path.join(self.image_dir, pid)

        cloth_front = self._load_cloth(os.path.join(base, data["cloth_front"]))
        cloth_back = self._load_cloth(os.path.join(base, data["cloth_back"]))

        candidate = self._load_model(os.path.join(base, data["candidate"]))
        front = self._load_model(os.path.join(base, data["front"]))
        side = self._load_model(os.path.join(base, data["side"]))
        back = self._load_model(os.path.join(base, data["back"]))

        grid = Image.new("RGB", (GRID_W, GRID_H), (128, 128, 128))
        # Top row
        grid.paste(cloth_front, (0, 0))
        grid.paste(cloth_back, (CLOTH_SIZE, 0))
        grid.paste(self.gray_id, (2 * CLOTH_SIZE, 0))
        # Bottom row
        grid.paste(candidate, (0, CLOTH_SIZE))
        grid.paste(front, (MODEL_W, CLOTH_SIZE))
        grid.paste(side, (2 * MODEL_W, CLOTH_SIZE))
        grid.paste(back, (3 * MODEL_W, CLOTH_SIZE))

        pixel_values = self.train_transforms(grid)

        # Mask: 0 for top row (conditioning), 1 for bottom row (to generate)
        mask = torch.zeros(1, GRID_H, GRID_W)
        mask[:, CLOTH_SIZE:, :] = 1.0

        return {
            "instance_images": pixel_values,
            "masks": mask,
            "sample_index": index,
        }


def collate_fn(examples):
    pixel_values = torch.stack([e["instance_images"] for e in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    masks = torch.stack([e["masks"] for e in examples])
    masks = masks.to(memory_format=torch.contiguous_format).float()

    sample_indices = [e["sample_index"] for e in examples]

    return {"pixel_values": pixel_values, "masks": masks, "sample_indices": sample_indices}


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder, tokenizer, max_sequence_length=512, prompt=None,
    num_images_per_prompt=1, device=None, text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_length=False, return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    dtype = text_encoder.module.dtype if hasattr(text_encoder, "module") else text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder, tokenizer, prompt: str, device=None,
    text_input_ids=None, num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True,
            return_overflowing_tokens=False, return_length=False, return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
    dtype = text_encoder.module.dtype if hasattr(text_encoder, "module") else text_encoder.dtype
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
    return prompt_embeds


def encode_prompt(
    text_encoders, tokenizers, prompt: str, max_sequence_length,
    device=None, num_images_per_prompt: int = 1, text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    dtype = text_encoders[0].module.dtype if hasattr(text_encoders[0], "module") else text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0], tokenizer=tokenizers[0], prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )
    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1], tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length, prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def main(args):
    global CLOTH_SIZE, MODEL_W, MODEL_H, GRID_W, GRID_H
    CLOTH_SIZE = args.cloth_size
    MODEL_W = CLOTH_SIZE * 3 // 4
    MODEL_H = CLOTH_SIZE
    GRID_W = 3 * CLOTH_SIZE
    GRID_H = 2 * CLOTH_SIZE

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision,
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        revision=args.revision, variant=args.variant,
    )

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2",
            "ff_context.net.0.proj", "ff_context.net.2",
            "proj_mlp", "proj_out",
            "norm.linear", "norm1.linear", "norm1_context.linear",
        ]

    transformer_lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian", target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in transformer.parameters())
    lora_debug_info = (
        f"========== LoRA Debug Info ==========\n"
        f"rank (r):        {args.rank}\n"
        f"lora_alpha:      {args.lora_alpha}\n"
        f"lora_dropout:    {args.lora_dropout}\n"
        f"target_modules:  {target_modules}\n"
        f"trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)\n"
        f"total params:     {total_params:,}\n"
        f"gradient_checkpointing: {args.gradient_checkpointing}\n"
        f"train_batch_size: {args.train_batch_size}\n"
        f"mixed_precision:  {args.mixed_precision}\n"
        f"=====================================\n"
    )
    logger.info(lora_debug_info)
    print(lora_debug_info)
    if accelerator.is_main_process:
        debug_log_path = os.path.join(args.output_dir, "lora_debug.log")
        with open(debug_log_path, "w") as _dbg:
            _dbg.write(lora_debug_info)
            _dbg.write(f"\nLoRA adapter config:\n{transformer_lora_config}\n")
            _dbg.write(f"\nTrainable parameter names:\n")
            for name, param in transformer.named_parameters():
                if param.requires_grad:
                    _dbg.write(f"  {name}: {list(param.shape)}\n")
        logger.info(f"LoRA debug info saved to {debug_log_path}")

    if args.resume_lora_path:
        logger.info(f"Loading pre-trained LoRA weights from {args.resume_lora_path}")
        lora_state_dict = FluxPipeline.lora_state_dict(args.resume_lora_path)
        transformer_state_dict = {
            k.replace("transformer.", ""): v
            for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")
        logger.info("Pre-trained LoRA weights loaded.")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            modules_to_save = {}
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["text_encoder"] = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                weights.pop()
            FluxPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxPipeline.lora_state_dict(input_dir)
        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.mixed_precision == "fp16":
        models = [transformer]
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include [adamW, prodigy]. "
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")
        optimizer_class = prodigyopt.Prodigy
        if args.learning_rate <= 0.1:
            logger.warning("Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0")
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    train_dataset = MultiViewDataset(
        jsonl_path=args.jsonl_path,
        image_dir=args.image_dir,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    # Pre-encode all unique prompts, then free text encoders
    all_prompts = train_dataset.prompts
    unique_prompts = sorted(set(all_prompts))
    prompt_to_idx = {p: i for i, p in enumerate(unique_prompts)}
    sample_to_prompt_idx = [prompt_to_idx[p] for p in all_prompts]

    logger.info(f"Pre-encoding {len(unique_prompts)} unique prompts (from {len(all_prompts)} samples)...")
    encode_bs = 64
    all_pe, all_ppe = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_prompts), encode_bs), desc="Pre-encoding prompts",
                      disable=not accelerator.is_local_main_process):
            batch_prompts = unique_prompts[i:i + encode_bs]
            pe, ppe, _ = encode_prompt(text_encoders, tokenizers, batch_prompts, args.max_sequence_length)
            all_pe.append(pe.cpu())
            all_ppe.append(ppe.cpu())

    cached_prompt_embeds = torch.cat(all_pe, dim=0)
    cached_pooled_prompt_embeds = torch.cat(all_ppe, dim=0)
    cached_text_ids = torch.zeros(cached_prompt_embeds.shape[1], 3).to(dtype=weight_dtype)
    logger.info(f"Cached prompt embeddings shape: {cached_prompt_embeds.shape}")

    del text_encoder_one, text_encoder_two, text_encoders, tokenizer_one, tokenizer_two, tokenizers
    free_memory()
    logger.info("Text encoders freed.")

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_name = "flux-dev-lora-multiview-caption"
        accelerator.init_trackers(tracker_name, config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                # Look up per-sample cached embeddings
                s_indices = batch["sample_indices"]
                p_indices = [sample_to_prompt_idx[i] for i in s_indices]
                prompt_embeds = cached_prompt_embeds[p_indices].to(accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = cached_pooled_prompt_embeds[p_indices].to(accelerator.device, dtype=weight_dtype)
                text_ids = cached_text_ids.to(accelerator.device)

                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                latent = vae.encode(pixel_values).latent_dist.sample()
                latent = (latent - vae_config_shift_factor) * vae_config_scaling_factor
                latent = latent.to(dtype=weight_dtype)

                masks = batch["masks"].to(dtype=vae.dtype)
                mask_pixel_values = pixel_values * (1 - masks)
                mask_latent = vae.encode(mask_pixel_values).latent_dist.sample()
                mask_latent = (mask_latent - vae_config_shift_factor) * vae_config_scaling_factor
                mask_latent = mask_latent.to(dtype=weight_dtype)
                packed_mask_input = FluxPipeline._pack_latents(
                    mask_latent, mask_latent.shape[0], mask_latent.shape[1],
                    mask_latent.shape[2], mask_latent.shape[3],
                )

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)
                height, width = [s // vae_scale_factor for s in mask_pixel_values.shape[-2:]]
                masks = masks.view(masks.shape[0], height, vae_scale_factor, width, vae_scale_factor)
                masks = masks.permute(0, 2, 4, 1, 3)
                masks = masks.reshape(masks.shape[0], vae_scale_factor * vae_scale_factor, height, width)
                masks = masks.to(dtype=weight_dtype)
                masks = FluxPipeline._pack_latents(
                    masks, masks.shape[0], vae_scale_factor * vae_scale_factor, height, width,
                )

                addition_latents = torch.cat([packed_mask_input, masks], dim=-1)

                noise = torch.randn_like(latent)
                bsz = latent.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=latent.device)

                sigmas = get_sigmas(timesteps, n_dim=latent.ndim, dtype=latent.dtype)
                noisy_model_input = (1.0 - sigmas) * latent + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=latent.shape[0],
                    num_channels_latents=latent.shape[1],
                    height=latent.shape[2],
                    width=latent.shape[3],
                )

                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    latent.shape[0],
                    latent.shape[2] // 2,
                    latent.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(latent.shape[0])
                else:
                    guidance = None

                model_input = torch.cat([packed_noisy_model_input, addition_latents], dim=2)
                model_pred = transformer(
                    hidden_states=model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=latent.shape[2] * vae_scale_factor,
                    width=latent.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                target = noise - latent

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            if accelerator.is_main_process:
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        modules_to_save = {}
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        modules_to_save["transformer"] = transformer

        FluxPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=None,
            **_collate_lora_metadata(modules_to_save),
        )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
