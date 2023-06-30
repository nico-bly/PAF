# %%
import wandb
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
from diffusers.models import AutoencoderKL
import json
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model7 = "models/dreambooth-SD-v1-5-4"
model6 = "models/dreambooth-SD-v1-5-3"
model5 = "models/dreambooth-SD-v1-5-2"
model4 = "models/dreambooth-SD-v1-5"
model3 = "digiplay/majicMIX_realistic_v6"
model2 = "runwayml/stable-diffusion-v1-5"
model = "SG161222/Realistic_Vision_V2.0"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
device = "cuda"


pipe = DiffusionPipeline.from_pretrained(
    model6, vae=vae, safety_checker=None, requires_safety_checker=False, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)


def get_all_prompts():
    f = open("prompts/prompts_SD_Humans.json")
    js = json.load(f)
    return js


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid

# %%


# prompts = get_all_prompts()
# prompt, neg = prompts["prompts_train"][0]

prompt = "photo of a person, style zwk, closeup"
neg = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, amputation"
# %%
with torch.autocast(device):
    img = pipe(prompt=prompt, negative_prompt=neg, width=512,
               height=512, num_inference_steps=40, guidance_scale=7).images[0]
plt.imshow(img)
# %%
imgs = []
with torch.autocast(device):
    for i in range(16):
        imgs.append(pipe(prompt=prompt, negative_prompt=neg, width=512,
                         height=512, guidance_scale=7.5).images[0])

plt.imshow(image_grid(imgs, 4, 4))

# %%

pipe = StableDiffusionPipeline.from_pretrained(
    model2, vae=vae, safety_checker=None, requires_safety_checker=False, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config)
pipe = pipe.to(device)
# %%

imgs = []
with torch.autocast(device):
    for i in range(16):
        imgs.append(pipe(prompt=prompt, negative_prompt=neg, width=512,
                         height=512, guidance_scale=7.5).images[0])

plt.imshow(image_grid(imgs, 4, 4))
