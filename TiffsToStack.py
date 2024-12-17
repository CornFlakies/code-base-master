# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:13:36 2024

@author: coena
"""

import os
import numpy as np
from tqdm import tqdm
from skimage import io
from PIL import Image

dir = "D:\\masterproject\\images\\dodecane_04122024\\Camera_1_C001H001S0001_C1S0001_20241204_172850"
dir = "D:\\masterproject\\images\\dodecane_04122024\\Camera_2_C002H001S0001_C2S0001_20241204_172850"

listfiles =[]

for img_files in os.listdir(dir):
    if img_files.endswith(".tif") :
        listfiles.append(img_files)


first_image = io.imread(os.path.join(dir,listfiles[0]))
first_image_pil = Image.fromarray(first_image)

# io.imshow(first_image)

first_image.shape

N = len(listfiles)
stack = []


for n, file in tqdm(enumerate(listfiles[1:])):
    stack.append(Image.fromarray(io.imread(os.path.join(dir,file))))

path_results = os.path.join(dir,'stack.tif')
first_image_pil.save(path_results, save_all=True, append_images=stack)