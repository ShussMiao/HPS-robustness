import os
import torch
import torch.nn.functional as F
import torchattacks
from PIL import Image
import numpy as np
from torch import nn

import hpsv2
#from Stable_Diffusion_image_generate import image_path
#from Stable_Diffusion_image_generate import prompt
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import hpsv2
from attack_function import attack_pgd
from hpsv2.src.open_clip import create_model_and_transforms
from torchvision import transforms

lower_limit, upper_limit = 0.0, 1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_path = r"D:\HPSv2-master\hpsv2\images\FLUX monkey gamer wearing red.png"
prompt = "monkey gamer wearing red - pink hoodie and a cap, aesthetic, stunning, highly detailed, smooth, soft focus highly detailed"

image = Image.open(image_path).convert("RGB")

# print score of original image
print("score of original image: ")
result1 = hpsv2.score(image_path, prompt, hps_version="v2.1")
scores1 = result1
print(scores1)

# add adversarial attack to original image
model, _, _ = create_model_and_transforms('ViT-L-14', pretrained='openai', precision='amp', device=device)
get_model=model.eval().cuda()  # Set the model to evaluation mode and move it to GPU

preprocess = transforms.Compose([
    transforms.ToTensor(),           # Convert the image to a tensor
])
image_copy = image.copy()
image_tensor = preprocess(image_copy).unsqueeze(0).cuda()  # Shape: (C, H, W)
print("image_tensor: ", image_tensor)
tokenizer = get_tokenizer('ViT-H-14')
text_tokens = tokenizer([prompt]).to(device=device, non_blocking=True)
target_tensor = torch.tensor([[1.0]], device='cuda')

delta = attack_pgd(
    prompter=None,
    model=get_model,
    add_prompter=None,
    criterion=nn.BCEWithLogitsLoss(),
    X=image_tensor,
    target=target_tensor,
    text_tokens=text_tokens,
    alpha=0.003,
    attack_iters=40,
    norm='l_inf',
    epsilon=0.03
)
print("delta: ", delta)

X_adv = image_tensor + delta
print("X_adv:", X_adv)
X_adv = torch.clamp(X_adv, lower_limit, upper_limit)
to_pil = transforms.ToPILImage()
X_adv_cpu = X_adv.squeeze(0).cpu()
X_adv_pil = to_pil(X_adv_cpu)
X_adv_pil.save(r"D:\HPSv2-master\hpsv2\images\attacked FLUX monkey gamer wearing red.png")

# print score of attacked image
print("score of attacked image: ")
adv_image_path = r"D:\HPSv2-master\hpsv2\images\attacked FLUX monkey gamer wearing red.png"
result2 = hpsv2.score(adv_image_path, prompt, hps_version="v2.1")
scores2 = result2
print(scores2)
