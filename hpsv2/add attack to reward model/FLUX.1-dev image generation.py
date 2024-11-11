import os
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")

prompt = "an otherworldly creature made of crystals, magnificent, majestic, highly intricate, realistic photography, incredibly detailed, ultra high resolution"
image = pipe(prompt).images[0]

pipe = pipe.to("cuda")
desired_size = (224, 224)
image = image.resize(desired_size)
save_dir = "D:\HPSv2-master\hpsv2\images"
image_path = os.path.join(save_dir, "FLUX an otherworldly creature.png")
image.save(image_path)