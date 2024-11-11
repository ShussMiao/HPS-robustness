import hpsv2


# original image
imgs_path = [r"D:\HPSv2-master\hpsv2\images\astronaut_riding_a_horse_on_mars.png"]
prompt = 'a photo of an astronaut riding a horse on mars.'
result1 = hpsv2.score(imgs_path, prompt, hps_version="v2.1")
print("original score:", result1)

# attacked image
attacked_imgs_path = [r"D:\HPSv2-master\hpsv2\images\attacked_astronaut_riding_a_horse_on_mars.png"]
result2 = hpsv2.score(attacked_imgs_path, prompt, hps_version="v2.1")
print("attacked score:", result2)