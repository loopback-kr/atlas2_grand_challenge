# Python Built-in libraries
import os, sys, json, csv, re, random, numpy as np
from os.path import join, basename, exists, splitext, dirname, isdir
from shutil import copy, copytree, rmtree
from glob import glob, iglob
from tqdm import tqdm, trange
from tqdm.contrib import tzip
# Medical imaging libraries
import nibabel as nib, SimpleITK as sitk
# BIDS
from bidsio import BIDSLoader
from settings import eval_settings


def check_integrity(src_dir: str, fname='*.nii.gz', chk_label_1=False):
    outlier_paths = []
    for path in tqdm(sorted(list(iglob(join(src_dir, '**', fname), recursive=True))), desc='CRC Check', colour='BLUE', dynamic_ncols=True):
        try:
            elements = np.unique(np.array(nib.load(path).dataobj))
            if chk_label_1:
                if len(elements) != 2:
                    raise Exception(f'Length is not 2; {elements}')
        except Exception as e:
            tqdm.write(path+': '+e.__str__())
            outlier_paths.append(path)
    return outlier_paths

def chk_CRC(src_dir:str, fname:str='*', recursive:bool=True):
    for path in tqdm(sorted(list(iglob(f'{src_dir}/{fname}', recursive={recursive}))), desc='CRC Checking'):
        try:
            _ = nib.load(path).get_fdata()
        except Exception as e:
            tqdm.write(path+': '+e.__str__())

def convert_to_bids_format(src_dir:str, dst_root_dir:str, fname:str='*.nii.gz'):
    for path in tqdm(sorted(glob(join(src_dir, fname))), desc=f'Convert to BIDS format', colour='green', dynamic_ncols=True):
        fname = basename(path).split('.nii.gz')[0]
        dst_fname = fname + '_ses-1_space-MNI152NLin2009aSym_label_L_desc-T1lesion_mask.nii.gz'
        dst_dir = join(dst_root_dir, fname, 'ses-1', 'anat')
        os.makedirs(dst_dir, exist_ok=True)
        fname = join(f'{fname}_{"ses-1_space-MNI152NLin2009aSym_label_L_desc-T1lesion_mask.nii.gz"}')

        tqdm.write(f'{src_dir}/{basename(path)} >> {dst_dir}/{dst_fname}')
        # Save as NiBabel file        
        nib.load(path).to_filename(join(dst_dir, dst_fname))

def union(src0_dir:str, src1_dir:str, dst_dir:str):
    os.makedirs(dst_dir, exist_ok=True)

    src0_paths = sorted(list(iglob(join(src0_dir, '*.nii.gz'), recursive=True)))
    src1_paths = sorted(list(iglob(join(src1_dir, '*.nii.gz'), recursive=True)))

    for src0_path, src1_path in tzip(src0_paths, src1_paths, desc='Union', colour='green', dynamic_ncols=True):
        img0 = nib.load(src0_path)
        img1 = nib.load(src1_path)

        if (img0.affine-img1.affine).all() == 0 and \
                (0 in np.unique(img0.dataobj) or 1 in np.unique(img0.dataobj)) and \
                (0 in np.unique(img1.dataobj) or 1 in np.unique(img1.dataobj)): # Inference 값이 0, 1로 구성되어 있는지 체크
            
            union = np.logical_or(img0.dataobj, img1.dataobj).astype(np.int8) #union
            img = nib.Nifti1Image(union, img0.affine)  # Save axis for data (just identity)
            
            img.to_filename(join(dst_dir, basename(src0_path)))
        else: raise Exception('Unknown affine data')


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        raise NotImplementedError('Pass the model_prediction_dir parameter explicitly.')
    else:
        src_dir = sys.argv[1]
        if isdir(src_dir):
            dst_dir = f'tmp/{src_dir}_BIDS'
            dst_zip_abspath = f'{os.environ["PWD"]}/out/{src_dir}.zip'
            os.makedirs(dirname(dst_zip_abspath), exist_ok=True)

            # union(union_src0_dir, union_src1_dir, union_dst_dir)
            convert_to_bids_format(src_dir, dst_dir)
            BIDSLoader.write_dataset_description(dst_dir, eval_settings['PredictionBIDSDerivativeName'][0])
            if os.getenv('CHECK_CRC', default=None) == 'True':
                check_integrity(dst_dir)
            
            os.system(f'cd "{dst_dir}" && zip -qr "{dst_zip_abspath}" *')
        else:
            raise Exception('Specify model output directory.')