# 读取nii_file
import numpy as np
import nibabel as nib
import cv2
import os

nii_file = ""
img_data = nib.load(nii_file)
img = img_data.get_fdata()
