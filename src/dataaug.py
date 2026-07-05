"""Post-processing images, e.g. reducing size, adding markers, etc.

Only works with local filesystem, not HF datasets."""
import abc
import itertools
import functools
import pathlib
import random
from typing import List, Union

import cv2
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import tqdm

import core
import expt
import util

import logging as lgr


TARGET_SEG_SUBDIR = "targ_segmts/"
DISTRACTOR_SEG_SUBDIR = "dist_segmts/"

AUGMENTED_SUBDIR = "augmented/"

CR_LEFT = "left"
CR_RIGHT = "right"
CR_TOP = "top"
CR_BOTTOM = "bottom"

HP_CUE = "HP"


class ImageEditor(abc.ABC):
    @abc.abstractmethod
    def process(self, img: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abc.abstractmethod
    def short_name(self) -> str:
        pass


class Cropper(ImageEditor):

    def __init__(self, side: str, pixels: Union[int, float]):
        self.side = side
        self.pixels = pixels

    def process(self, img: np.ndarray, **kwargs) -> np.ndarray:
        x0, y0, x1, y1 = 0, 0, img.shape[1], img.shape[0]
        if self.side == CR_LEFT:
            x0 = self.int_pixels(self.pixels, img.shape[1])
        elif self.side == CR_RIGHT:
            x1 -= self.int_pixels(self.pixels, img.shape[1])
        elif self.side == CR_TOP:
            y0 = self.int_pixels(self.pixels, img.shape[0])
        elif self.side == CR_BOTTOM:
            y1 -= self.int_pixels(self.pixels, img.shape[0])
        else:
            raise ValueError(f"Invalid side value: {self.side}")

        return img[y0:y1, x0:x1]

    def short_name(self) -> str:
        return f"cr{self.side[0]}"

    @staticmethod
    def int_pixels(pixels, total):
        if pixels > 1:
            return int(pixels)

        assert 0 < pixels, f"{pixels} must be in (0, 1)"
        return round(total * pixels)


class MaxLengthRescaler(ImageEditor):

    def __init__(self, pixels=1024):
        self.pixels = pixels

    def process(self, img: np.ndarray, **kwargs) -> np.ndarray:
        scaler = self.pixels / max(img.shape)
        if scaler >= 1:
            return img

        new_w = round(img.shape[1] * scaler)
        new_h = round(img.shape[0] * scaler)
        return cv2.resize(img, (new_w, new_h))

    def short_name(self) -> str:
        return "ml"


class OpencvMarker(ImageEditor):
    def __init__(self, color: str, shape: str, size=10):
        self.color = color
        self.shape = shape
        self.size = size

    def process(self, img: np.ndarray, segmt: np.ndarray=None) -> np.ndarray:
        img = img[:, :, :3]  # for png
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)

        for x, y in self._list_seg_centers(segmt):
            img = self.mark(img, x, y)

        return img

    def short_name(self) -> str:
        return f"mk{self.color}{self.shape[0]}"

    def _list_seg_centers(self, segmt: np.ndarray):
        """List the center (of mass) for potentially multiple segments."""
        centers = []
        for val in np.unique(segmt):
            if val == 0:
                continue
            temp_mask = np.zeros_like(segmt)
            temp_mask[segmt == val] = val
            center = util.calc_mask_center(temp_mask)
            centers.append(center)
        return centers

    def mark(self, img_: np.ndarray, x, y):
        x, y = round(x), round(y)
        bgr_color = self._to_bgr(self.color)
        thickness = -1  # -1 means filled

        if self.shape == "circle":
            radius = self.size // 2
            img_ = cv2.circle(img_.copy(), (x, y), radius, bgr_color, thickness)

        elif self.shape == "square":
            top_left = (x - self.size//2, y - self.size//2)
            bottom_right = (x + self.size//2, y + self.size//2)
            img_ = cv2.rectangle(img_.copy(), top_left, bottom_right, bgr_color, thickness)

        else:
            raise ValueError(f"Unsupported shape: {self.shape}")

        return img_

    @staticmethod
    def _to_bgr(color_name: str):
        """Because image loaded was not converted to bgr, so bgr conversion is ignored"""
        rgb_color = np.array(mcolors.to_rgb(color_name)) * 255  # RGB [0-255]
        rgb_color = rgb_color.astype(np.uint8).tolist()
        # bgr_color = rgb_color[::-1].astype(np.uint8).tolist()  # Convert to BGR
        return rgb_color


class UndergroundFixer(ImageEditor):
    """Against underground plane reflection."""

    CUES_TO_SKIP = [HP_CUE]  # matching imggen.main

    def __init__(self):
        pass

    def process(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """For any image w/o HP cue, making the bottom half black-ish."""
        img_h = img.shape[0]
        img[img_h//2:, :, :3] = 0  # excl alpha channel (3+ channels assumed)
        return img

    def short_name(self) -> str:
        return 'uf'


class ImageAugmenter:
    SUFFIX_STR = "-augmented-"

    def __init__(self, editors: List[ImageEditor]):
        self.editors = editors

    def augment(self, img: np.ndarray, **kwargs) -> np.ndarray:
        img_ = img
        for editor in self.editors:
            img_ = editor.process(img_, **kwargs)
        return img_

    def get_suffix(self):
        editor_desc = "-".join(editor.short_name() for editor in self.editors)
        return f'{self.SUFFIX_STR}{editor_desc}'


class BaseArrayLoader(abc.ABC):
    def __init__(self, args, img_set: expt.BaseImageSet):
        self.args = args
        self.img_set = img_set

    @abc.abstractmethod
    def load_img(self, img_name: str) -> np.ndarray:
        pass

    @abc.abstractmethod
    def load_targ_seg(self, img_name: str) -> np.ndarray:
        pass

    @abc.abstractmethod
    def load_dist_seg(self, img_name: str) -> np.ndarray:
        pass


class FileSysArrayLoader(BaseArrayLoader):
    def load_img(self, img_name: str) -> np.ndarray:
        img_array = self.img_set.get_image(img_name).to_numpy()
        return img_array

    def load_targ_seg(self, img_name: str) -> np.ndarray:
        img_path = self.img_set.get_image_path(img_name)
        tsg_path = pathlib.Path(img_path).parent.parent.parent / TARGET_SEG_SUBDIR / 'orig' / img_name
        return cv2.imread(str(tsg_path), cv2.IMREAD_GRAYSCALE)

    def load_dist_seg(self, img_name: str) -> np.ndarray:
        img_path = self.img_set.get_image_path(img_name)
        dsg_path = pathlib.Path(img_path).parent.parent.parent / DISTRACTOR_SEG_SUBDIR / 'orig' / img_name
        return cv2.imread(str(dsg_path), cv2.IMREAD_GRAYSCALE)


class HuggingFaceArrayLoader(BaseArrayLoader):
    def __init__(self, args, img_set: expt._HuggingFaceImageSet):
        super().__init__(args, img_set)

        assert isinstance(self.img_set, expt._HuggingFaceImageSet)
        self.hf_dataset = self.img_set.ds

    def load_img(self, img_name: str) -> np.ndarray:
        sample = self._get_sample(img_name)
        img = sample[self.img_set.MAIN_IMAGE_COLUMN]
        return np.array(img)

    def load_targ_seg(self, img_name: str) -> np.ndarray:
        sample = self._get_sample(img_name)
        seg = sample['targ_seg']
        return np.array(seg)

    def load_dist_seg(self, img_name: str) -> np.ndarray:
        sample = self._get_sample(img_name)
        seg = sample['dist_seg']
        return np.array(seg)

    @functools.lru_cache()
    def _get_sample(self, img_name):
        return util.df_dataset_get_sample(self.hf_dataset, img_name)


def get_array_loader(args, img_set: expt.BaseImageSet) -> BaseArrayLoader:
    if isinstance(img_set, expt._DirectoryImageSet):
        return FileSysArrayLoader(args, img_set)
    elif isinstance(img_set, expt._HuggingFaceImageSet):
        return HuggingFaceArrayLoader(args, img_set)
    else:
        raise ValueError(f"Unknown image set type: {type(img_set)}")


def main(args):
    img_set = expt.get_image_set(args.image_set, **args.imset_kwargs)

    if args.mark:
        mark_images(img_set, args)
    else:
        augment_images(img_set, args)


def augment_images(img_set: expt.BaseImageSet, args):
    """Note: only works with local file systems, with the following dir structure:

    data/
        <image_set>/  # e.g. kubric_scenes/one_cue/
            images/
                orig/*.png
                augmented/  # <-- output dir of this function
    """
    if isinstance(img_set, expt._HuggingFaceImageSet):
        raise NotImplementedError("Cannot augment images loaded from HF.")

    editors = []
    if args.underground:
        editors += [UndergroundFixer()]
    editors += [
        Cropper(side=s,
                pixels=args.cr_pixels if s in {CR_LEFT, CR_RIGHT}
                else args.cr_pixels * 2)   # orig imgs have more vertical space
        for s in args.cr_sides
    ]
    editors += [MaxLengthRescaler()]
    img_aug = ImageAugmenter(editors=editors)

    img_names = list(_filtered_image_names(args, img_set))
    for img_name in tqdm.tqdm(img_names):
        img_path = img_set.get_image_path(img_name)
        out_path = _format_out_path(img_path, img_aug.get_suffix())

        if out_path.exists() and not args.regen:
            continue

        augmented_img = _augment_img(img_aug, img_path)
        _save_img(out_path, augmented_img)


def mark_images(img_set: expt.BaseImageSet, args):
    """Output dir structure:

    data/
        <image_set>/  # e.g. kubric_scenes/one_cue/
            images/
                orig/*.png
                augmented/*.png
                marked/  # <-- output dir of this function
            targ_segmts/
                orig/*.png
            dist_segmts/
                orig/*.png
    """
    array_loader = get_array_loader(args, img_set)

    targ_aug = ImageAugmenter(editors=[
        OpencvMarker('r', 'circle', size=args.mark_size),
    ])

    dist_editors = [
        OpencvMarker('b', 'square', size=args.mark_size),
        MaxLengthRescaler(),
    ]
    if args.underground:
        dist_editors.append(UndergroundFixer())
    dist_aug = ImageAugmenter(editors=dist_editors)

    img_names = list(_filtered_image_names(args, img_set))
    for img_name in tqdm.tqdm(img_names):
        img_path = img_set.get_image_path(img_name)
        suffix = targ_aug.get_suffix()+dist_aug.get_suffix()
        out_path = _format_out_path(img_path, suffix, subdir="marked/")

        if out_path.exists() and not args.regen:
            continue

        img = array_loader.load_img(img_name)
        targ_seg = array_loader.load_targ_seg(img_name)
        dist_seg = array_loader.load_dist_seg(img_name)

        augmented_img = targ_aug.augment(img, segmt=targ_seg)
        augmented_img = dist_aug.augment(augmented_img, segmt=dist_seg)
        _save_img(out_path, augmented_img)

    _post_process(img_set)


def _filtered_image_names(args, img_set: expt.BaseImageSet):
    img_names = img_set.list_image_names()
    for img_name in img_names:
        if args.hp and HP_CUE not in img_name:
            continue

        if args.underground and any(cue in img_name
                                    for cue in UndergroundFixer.CUES_TO_SKIP):
            continue

        yield img_name


def _format_out_path(path, suffix, subdir=AUGMENTED_SUBDIR) -> pathlib.Path:
    augmented_path = util.format_suffix(path, suffix)
    augmented_path = util.format_subdir(augmented_path, subdir, sibling=True)
    return augmented_path


def _augment_img(img_aug, path):
    img = core.FileSysImage(path, None).to_numpy()
    augmented_img = img_aug.augment(img)
    return augmented_img


def _save_img(path: pathlib.Path, img):
    try:
        plt.imsave(path, img)
    except FileNotFoundError:
        path.parent.mkdir(parents=True)
        plt.imsave(path, img)


def _post_process(img_set: expt.BaseImageSet):
    if isinstance(img_set, expt._HuggingFaceImageSet):
        output_dir = util.DEFAULT_DATA_PATH/img_set.SUBDIR_NAME
        img_set_root = output_dir.parent.parent
        img_set.info.to_json(img_set_root/expt.METADATA_FILENAME, lines=True)


if __name__ == "__main__":
    parser = util.get_argparser()
    parser.add_argument("--cr_sides", help="side to crop", required=False, nargs="*",
                        choices=[CR_LEFT, CR_RIGHT, CR_TOP, CR_BOTTOM], default=[])
    parser.add_argument("--cr_pixels", help="pixels or proportion to crop",
                        default=0.1, type=float)

    parser.add_argument("--underground", help="fix underground reflection",
                        action="store_true", default=False)
    parser.add_argument("--hp", help="only run some specific aug for HP cue",
                        action="store_true", default=False)

    parser.add_argument("--mark", help="mark objects for easier reference",
                        action="store_true", default=False)
    parser.add_argument("--mark_size", help="width or diameter of markers",
                        default=20, type=int)

    parser.add_argument("--regen", help="regenerate even if exists",
                        action="store_true", default=False)

    args = parser.parse_args()


    main(args)
