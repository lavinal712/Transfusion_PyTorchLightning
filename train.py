import argparse
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import AutoTokenizer

from diffusion import create_diffusion
from models.transfusion import Transfusion_models


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def parse_args(input_args=None):
    parser = argparse.ArgumentParser("Train Transfusion")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(Transfusion_models.keys()), default="Transfusion_G")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--text_length", type=int, default=1024)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    accelerator = Accelerator()
    device = accelerator.device
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    requires_grad(vae, False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.unk_token
    vocab_size = tokenizer.vocab_size
    model = Transfusion_models[args.model](
        input_size=latent_size,
        block_size=args.text_length,
        vocab_size=vocab_size,
    )
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = model.to(device)
    diffusion = create_diffusion(timestep_respacing="")
    with open("data/imagenet1000_clsidx_to_labels.txt", "r") as f:
        id2label = eval(f.read())
    if accelerator.is_main_process:
        logger.info(f"Transfusion Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    
    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()
    model, opt, loader = accelerator.prepare(model, opt, loader)

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                y = list(map(lambda id: id2label[int(id)], y))
                y = tokenizer(y, padding="max_length", max_length=args.text_length, return_tensors="pt")["input_ids"].to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict, logits = diffusion.training_losses(model, x, t, model_kwargs)
            diff_loss = loss_dict["loss"].mean()
            lm_loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            loss = lm_loss + diff_loss
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item()
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()

    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
