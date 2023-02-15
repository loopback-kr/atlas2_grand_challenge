import os, numpy as np, nibabel as nib, SimpleITK as sitk, json, csv, re, random, tqdm
from tqdm.contrib import tzip
from os.path import join, basename, exists, splitext, dirname
from glob import glob, iglob
from shutil import copy

A_PATH = os.getenv('A_PATH', default=None)
B_PATH = os.getenv('B_PATH', default=None)
DST_PATH = os.getenv('DST_PATH', default=None)

a_paths = sorted(glob(join(A_PATH, '*.nii.gz')))
b_paths = sorted(glob(join(B_PATH, '*.nii.gz')))
os.makedirs(DST_PATH, exist_ok=True)

for a_path, b_path in tzip(a_paths, b_paths, dynamic_ncols=True, desc=f'{DST_PATH}'):
    if basename(a_path) != basename(b_path):
        tqdm.write(f'Not matched: {basename(a_path)}, {basename(b_path)}')
    else:
        a = sitk.ReadImage(a_path)
        b = sitk.ReadImage(b_path)

        a_img = sitk.GetArrayFromImage(a)
        b_img = sitk.GetArrayFromImage(b)

        union_img = np.logical_or(a_img, b_img).astype(np.uint8)

        union = sitk.GetImageFromArray(union_img)
        union.CopyInformation(a)

        sitk.WriteImage(union, join(DST_PATH, basename(a_path)))

        # np_union = sitk.GetArrayFromImage(union).transpose(2, 1, 0)
        # if os.getenv('SAVE_ALSO_NPY', default=None) == 'true':
            # np.savez_compressed(join(DST_PATH, f'{basename(a_path).replace(".nii.gz", ".npz")}'), data=np_union.astype(np.float32))
        # if os.getenv('COPY_PKL', default=None) == 'true':
            # src_dir = dirname(a_path)
            # src_filename = basename(a_path).replace('.nii.gz', '.pkl')
            # copy(join(src_dir, src_filename), join(DST_PATH, src_filename))