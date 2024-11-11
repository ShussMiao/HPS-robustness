import os
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "monkey gamer wearing red - pink hoodie and a cap, aesthetic, stunning, highly detailed, smooth, soft focus highly detailed"
image = pipe(prompt).images[0]
desired_size = (224, 224)
image = image.resize(desired_size)
save_dir = "D:\HPSv2-master\hpsv2\images"
image_path = os.path.join(save_dir, "monkey gamer wearing red.png")
image.save(image_path)