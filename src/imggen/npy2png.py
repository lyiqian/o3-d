"""Helpers converting .npy files to standard .png for segmentation masks."""
import argparse
import multiprocessing
import pathlib

import tqdm
import numpy as np
from PIL import Image


N_PROC = 5


def convert(npy_path):
    npy_path = str(npy_path)
    try:
        mask = np.load(npy_path)
    except Exception as e:
        print(f"Failed to load npy file: {repr(e)}")

    mask = mask * 255 // mask.max()  # seg_id -> grey scale, for easier visualization
    mask_uint8 = mask.astype(np.uint8)
    mask_2d = mask_uint8.squeeze()
    mask_img = Image.fromarray(mask_2d)

    try:
        png_path = npy_path.replace(".npy", ".png")
        mask_img.save(png_path)
    except Exception as e:
        print(f"Failed to write png file: {repr(e)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dirpath')

    args = parser.parse_args()
    dirpath = pathlib.Path(args.dirpath)
    npy_paths = list(dirpath.glob("*.npy"))

    with multiprocessing.Pool(processes=N_PROC) as executor:
        for npy_path in tqdm.tqdm(npy_paths, total=len(npy_paths)):
            # print(npy_path)
            executor.apply(convert, (npy_path,))


if __name__ == "__main__":
    main()