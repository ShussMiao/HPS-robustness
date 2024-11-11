# import hpsv2
# from datasets import load_dataset, Image  #benchmark/benchmark_imgs
#
# dataset = load_dataset("D:\HPSv2-master\CM.tar\CM")
# #dataset = load_dataset("D:\HPSv2-master\CM.tar\CM", split="train")
# print(hpsv2.get_available_models()) # Get models that have access to data
# hpsv2.evaluate_benchmark('CM')

import huggingface_hub
import hpsv2

print(hpsv2.get_available_models()) # Get models that have access to data
hpsv2.evaluate_benchmark('<model_name>')