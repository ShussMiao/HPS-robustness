import os
import torch
import torch.nn.functional as F
import torchattacks
from PIL import Image
import numpy as np
import hpsv2
from Stable_Diffusion_image_generate import pipe
from Stable_Diffusion_image_generate import image_path
from Stable_Diffusion_image_generate import prompt
from hpsv2.src.open_clip import create_model_and_transforms
from pgd_on_HPSv2 import PGDReward
import hpsv2
from attack_function import attack_pgd
from hpsv2.src.open_clip import create_model_and_transforms
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image = Image.open(image_path).convert("RGB")

# print score of original image
print("score of original image: ")
result1 = hpsv2.score(image_path, prompt, hps_version="v2.1")
scores1 = result1
print(scores1)

# add adversarial attack to original image
model, _, _ = create_model_and_transforms('ViT-L-14', pretrained='openai', precision='amp', device=device)
print(model)