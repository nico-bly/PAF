import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import json
from tqdm import tqdm

def get_all_prompts():
    f = open("prompts_SD.json")
    js = json.load(f)
    return js

prompts = get_all_prompts()
dest = "D_Trump_Fake"

model_id = "CompVis/stable-diffusion-v1-3"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker = None, requires_safety_checker = False, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)


print("[+] Generating Train Set")
for i, prompt in tqdm(enumerate(prompts["prompts_train"])):
    for j in range(25):
        img = pipe(prompt, width=512, height=512, guidance_scale=7.5).images[0]
        img.save(dest+"/trump_prompt_train_i_{}_j_{}.png".format(i,j))

print("[+] Generating Test Set")
for i, prompt in tqdm(enumerate(prompts["prompts_test"])):
    for j in range(25):
        img = pipe(prompt, width=512, height=512, guidance_scale=7.5).images[0]
        img.save(dest+"/trump_prompt_test_i_{}_j_{}.png".format(i,j))