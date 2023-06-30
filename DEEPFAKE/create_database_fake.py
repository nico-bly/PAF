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
    f = open("prompts_SD_Humans_V2.json")
    js = json.load(f)
    return js


prompts = get_all_prompts()

model = "SG161222/Realistic_Vision_V2.0"
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
    for i in range(17):
        if i < 11:
            folder = "data_human_v2/train/Human_Fake/"
        elif i < 14:
            folder = "data_human_v2/val/Human_Fake/"
        else:
            folder = "data_human_v2/test/Human_Fake/"
        with torch.autocast(device):
            img = pipe(prompt=prompt, negative_prompt=neg_prompt,
                       width=512, height=512, num_inference_steps=25, guidance_scale=7.5).images[0]
        img.save(folder + str(count) + ".png")
        count += 1
        wandb.log({"count": count})

wandb.alert(
    title="Database Created",
    text="3672 in Total"
)

wandb.finish()

# print("[+] Generating Train Set")
# for i, prompt in tqdm(enumerate(prompts["prompts_train"])):
#     for j in range(126):
#         img = pipe(prompt, width=512, height=512, guidance_scale=7.5).images[0]
#         img.save(dest+"/train/D_Trump_Fake/trump_prompt_train_i_{}_j_{}.png".format(i,j))

# print("[+] Generating Test Set")
# for i, prompt in tqdm(enumerate(prompts["prompts_test"])):
#     for j in range(126):
#         img = pipe(prompt, width=512, height=512, guidance_scale=7.5).images[0]
#         img.save(dest+"/test/D_Trump_Fake/trump_prompt_test_i_{}_j_{}.png".format(i,j))

# print("[+] Generating Val Set")
# for i, prompt in tqdm(enumerate(prompts["prompts_val"])):
#     for j in range(126):
#         img = pipe(prompt, width=512, height=512, guidance_scale=7.5).images[0]
#         img.save(dest+"/val/D_Trump_Fake/trump_prompt_val_i_{}_j_{}.png".format(i,j))
