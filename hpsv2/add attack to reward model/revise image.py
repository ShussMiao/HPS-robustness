import os

from PIL import Image

image = Image.open(r"C:\Users\29238\OneDrive\Desktop\6bc29b10-9ce6-4d33-a9b9-5a11ce881a8d.png").convert("RGB")
desired_size = (224, 224)
image = image.resize(desired_size)
save_dir = "D:\HPSv2-master\hpsv2\images"
image_path = os.path.join(save_dir, "FLUX monkey gamer wearing red.png")
image.save(image_path)