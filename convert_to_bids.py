import os, numpy as np, nibabel as nib
from os.path import join, basename, dirname, exists
from tqdm import tqdm
from glob import glob, iglob
from bidsio import BIDSLoader
from settings import eval_settings
from tqdm.contrib import tzip

# Define global vars
model_prediction_dir = os.getenv('MODEL_PREDICTION_DIR', default=None)
BIDS_formatted_dir = 'BIDS_' + model_prediction_dir.split('/')[-1]
zip_filename = f'{BIDS_formatted_dir}.zip'


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

def convert_to_BIDS(src_dir:str, dst_dir:str, fname:str='*.nii.gz'):
    for path in tqdm(sorted(glob(join(src_dir, fname))), desc='Convert to BIDS', colour='green', dynamic_ncols=True):
        # Directory configuration
        tqdm.write(path)
        first_dir = 'sub-r'+basename(path)[3:6]+'s'+basename(path)[7:10]
        target_dir = join(dst_dir, first_dir, 'ses-1', 'anat')
        os.makedirs(target_dir, exist_ok=True)
        fname = join(f'{first_dir}_{"ses-1"}_{"space-MNI152NLin2009aSym_label_L_desc-T1lesion_mask.nii.gz"}')

        # Save as NiBabel file        
        nib.load(path).to_filename(join(target_dir, fname))

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
    if exists(zip_filename):
        os.remove(zip_filename)
    # union(union_src0_dir, union_src1_dir, union_dst_dir)
    convert_to_BIDS(model_prediction_dir, BIDS_formatted_dir)
    BIDSLoader.write_dataset_description(BIDS_formatted_dir, eval_settings['PredictionBIDSDerivativeName'][0])
    if os.getenv('CHECK_CRC', default=None) == 'True':
        check_integrity(BIDS_formatted_dir)
    os.system(f'cd "{BIDS_formatted_dir}" && zip -qr "../{zip_filename}" *')
    os.system(f'rm -rf "{BIDS_formatted_dir}"')