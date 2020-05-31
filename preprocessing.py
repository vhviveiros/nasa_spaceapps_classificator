# %%Imports
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import job
from image_segmentation import segmentate_images as seg
import pandas as pd
from image import Image, ImageGenerator, ImageSaver
from image_processor import process

covid_path = os.path.join('dataset/covid')
covid_masks_path = os.path.join('cov_masks')

non_covid_path = os.path.join('dataset/normal')
non_covid_masks_path = os.path.join('non_cov_masks')


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# %%Segmentation
check_folder(covid_masks_path)
check_folder(non_covid_masks_path)

seg(covid_path, 'cov_masks')
seg(non_covid_path, 'non_cov_masks')

# %%Read images
generator = ImageGenerator()

covid_images, covid_masks, non_covid_images, non_covid_masks = generator.generate_image_data(
    covid_path,
    covid_masks_path,
    non_covid_path,
    non_covid_masks_path
)

# %%Processing
cov_processed, non_cov_processed = process(
    covid_images,
    covid_masks,
    non_covid_images,
    non_covid_masks)

# %%Saving
cov_save_path = os.path.join('cov_processed')
non_cov_save_path = os.path.join('non_cov_processed')

check_folder(cov_save_path)
check_folder(non_cov_save_path)

ImageSaver(list(cov_processed.result())).save_to(cov_save_path)
ImageSaver(list(non_cov_processed.result())).save_to(non_cov_save_path)


# %%
