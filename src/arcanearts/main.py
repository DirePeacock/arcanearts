import os
import pdb
import pathlib
from PIL import Image, ImageDraw
import cv2
import numpy as np
from IPython.display import HTML
from base64 import b64encode

import torch
from torch import autocast
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
import huggingface_hub
from huggingface_hub.commands import huggingface_cli as hf_cli

from .localserver import start_server, close_server

# from google.colab import output

"""
I'm starting from this with zero knowledge of this stuff at the first YT result for stable diffusion code lol
copying from the notebook off of that video
https://colab.research.google.com/drive/1_kbRZPTjnFgViPrmGcUsaszEdYa8XTpq?usp=sharing#scrollTo=Mb-u1yDAJ23R
"""
device = "cuda"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = text_encoder.to(device)

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=True)
unet = unet.to(device)

# 4. Create a scheduler for inference
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=500
)


def get_hf_token():
    hf_file = pathlib.Path(os.getenv("HOME")) / ".hf_token.txt"
    line = None
    with open(hf_file, "r") as hf_file:
        line = str(hf_file.readline())
    if line is None:
        line = input(
            "please input a hugging face token from \n\thttps://huggingface.co/settings/tokens\nor put it in ~/.hf_token.txt"
        )
    return line


def huggingface_login():
    token = get_hf_token()
    login_cmd = f"echo '{token}' | huggingface-cli login"
    # print(login_cmd)
    # os.system(login_cmd)
    # print("\nwe're in\n")

    huggingface_hub.notebook_login()
    # print("\nnotebook login\n")


def do_main(args):
    # make sure you're logged in with `huggingface-cli login`
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True
    )
    pipe = pipe.to(device)
    prompt = "Upright squid"
    img = prompt_to_img(prompt, num_inference_steps=30)[0]

    prompt = "Squidward"
    img_latents = encode_img_latents([img])
    img = prompt_to_img(prompt, num_inference_steps=30, latents=img_latents, start_step=20)[0]
    dump_image(image, prompt=prompt)
    return

    prompt = "Cute shiba inu dog"
    with autocast(device):
        image = pipe(prompt)["sample"][0]
    if image is None:
        print("no img returned from pipe")
        return image

    dump_image(image, prompt=prompt)


def dump_image(image, prompt):
    "save img, and open it in ff"
    prompt_name = prompt
    file_name = "bruh.jpg"
    image.save(file_name)


def get_text_embeds(prompt):
    # Tokenize text and get embeddings
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    # Do the same for unconditional embeddings
    uncond_input = tokenizer(
        [""] * len(prompt), padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Cat for final embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings


def encode_img_latents(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]

    img_arr = np.stack([np.array(img) for img in imgs], axis=0)
    img_arr = img_arr / 255.0
    img_arr = torch.from_numpy(img_arr).float().permute(0, 3, 1, 2)
    img_arr = 2 * (img_arr - 0.5)

    latent_dists = vae.encode(img_arr.to(device))
    latent_samples = latent_dists.sample()
    latent_samples *= 0.18215

    return latent_samples


def produce_latents(
    text_embeddings,
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    latents=None,
    return_all_latents=False,
    start_step=10,
):
    if latents is None:
        latents = torch.randn((text_embeddings.shape[0] // 2, unet.in_channels, height // 8, width // 8))
    latents = latents.to(device)

    scheduler.set_timesteps(num_inference_steps)
    if start_step > 0:
        start_timestep = scheduler.timesteps[start_step]
        start_timesteps = start_timestep.repeat(latents.shape[0]).long()

        noise = torch.randn_like(latents)
        latents = scheduler.add_noise(latents, noise, start_timesteps)

    latent_hist = [latents]
    with autocast("cuda"):
        for i, t in tqdm(enumerate(scheduler.timesteps[start_step:])):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents)["prev_sample"]
            latent_hist.append(latents)

    if not return_all_latents:
        return latents

    all_latents = torch.cat(latent_hist, dim=0)
    return all_latents


def prompt_to_img(
    prompts,
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    latents=None,
    return_all_latents=False,
    batch_size=2,
    start_step=0,
):
    if isinstance(prompts, str):
        prompts = [prompts]

    # Prompts -> text embeds
    text_embeds = get_text_embeds(prompts)

    # Text embeds -> img latents
    latents = produce_latents(
        text_embeds,
        height=height,
        width=width,
        latents=latents,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        return_all_latents=return_all_latents,
        start_step=start_step,
    )

    # Img latents -> imgs
    all_imgs = []
    for i in tqdm(range(0, len(latents), batch_size)):
        imgs = decode_img_latents(latents[i : i + batch_size])
        all_imgs.extend(imgs)

    return all_imgs


def prompt_to_img(
    prompts,
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    latents=None,
    return_all_latents=False,
    batch_size=2,
    start_step=0,
):
    if isinstance(prompts, str):
        prompts = [prompts]

    # Prompts -> text embeds
    text_embeds = get_text_embeds(prompts)

    # Text embeds -> img latents
    latents = produce_latents(
        text_embeds,
        height=height,
        width=width,
        latents=latents,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        return_all_latents=return_all_latents,
        start_step=start_step,
    )

    # Img latents -> imgs
    all_imgs = []
    for i in tqdm(range(0, len(latents), batch_size)):
        imgs = decode_img_latents(latents[i : i + batch_size])
        all_imgs.extend(imgs)


def main(args):
    start_server()
    try:
        huggingface_login()

        do_main(args)
    except Exception as e:
        print(e)

    close_server()


if __name__ == "__main__":
    main(0)
