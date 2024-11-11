import os
import hpsv2
from transformers import pipeline
import torch
from diffusers import StableDiffusionPipeline
# Get benchmark prompts (<style> = all, anime, concept-art, paintings, photo)
all_prompts = hpsv2.benchmark_prompts('all')

# Iterate over the benchmark prompts to generate images


pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
#image = pipe(prompt).images[0]


for style, prompts in all_prompts.items():
    for idx, prompt in enumerate(prompts):
        #image = TextToImageModel(prompt)
        image = pipe(prompt)
        # TextToImageModel is the model you want to evaluate
        image.save(os.path.join(r"D:\HPSv2-master\pictures\generated images", style, f"{idx:05d}.jpg"))
        # <image_path> is the folder path to store generated images, as the input of hpsv2.evaluate().







# import torch
# from diffusers import StableDiffusionPipeline
#
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
#
# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]
#
# image.save("astronaut riding a horse on mars.png")