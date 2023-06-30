from PIL import Image
from tqdm import tqdm
import json
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
from diffusers.models import AutoencoderKL
from torch import autocast
import torch
import wandb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def get_all_prompts():
    f = open("prompts/prompts_SD_Humans_V4.json")
    js = json.load(f)
    return js


prompts = get_all_prompts()
# ethnicity = "old man with american ethinicity, clothed"

# prompt = "A {}, (detailed skin:1.2), dslr, film grain, Fujifilm XT3".format(ethnicity)
# neg = "black and white, disfigured, art, cgi"

model = "models/dreambooth-SD-v1-5-4"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
device = "cuda"

pipe = DiffusionPipeline.from_pretrained(
    model, vae=vae, safety_checker=None, requires_safety_checker=False, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

wandb.init(
    project="deepfake-paf",

    config={
        "n_tot": 3672
    }
)

count = 0
for data in prompts["prompts_train"]:
    prompt, neg_prompt = data
    for i in range(25):
        if i < 17:
            folder = "data_human_v6/train/Human_Fake/"
        elif i < 21:
            folder = "data_human_v6/val/Human_Fake/"
        else:
            folder = "data_human_v6/test/Human_Fake/"
        with torch.autocast(device):
            img = pipe(prompt=prompt, negative_prompt=neg_prompt,
                       width=512, height=512, num_inference_steps=40, guidance_scale=7.5).images[0]
        img.save(folder + str(count) + ".png")
        count += 1
        wandb.log({"count": count})

wandb.alert(
    title="Database Created",
    text="??? in Total"
)

wandb.finish()
