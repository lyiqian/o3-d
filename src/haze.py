"""Adding artificial haze to images for Saturation cue."""

import pathlib
# from diffusers.utils import load_image
from PIL import Image
from tqdm import tqdm
# from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np
# import torch

# import expt  # only using `apply_haze_eq` function here; no need to import
import util


DEPTH_EST_HF_MODEL = "Intel/zoedepth-nyu-kitti"


def main(args):
    image_set = expt.get_image_set(args.image_set)
    image_names = image_set.list_image_names()
    if util.DEBUG: image_names = image_names[:100]  # for testing

    # zoedepth for metric depth estimation
    image_processor = AutoImageProcessor.from_pretrained(DEPTH_EST_HF_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_EST_HF_MODEL)

    for image_name in tqdm(image_names):
        image_path = image_set.get_image_path(image_name)
        image = load_image(image_path)

        hazed_a = add_haze(image_processor, model, image)
        hazed_img = Image.fromarray(hazed_a.astype("uint8"))

        haze_subdir = "zoedepth-haze"
        result_dirpath = util.DEFAULT_DATA_PATH / "results" / haze_subdir / args.image_set
        result_dirpath.mkdir(parents=True, exist_ok=True)
        prefixed_image_name = f"{util.timestamp_str()}_{image_name}"
        hazed_img.save(str(result_dirpath/prefixed_image_name))


def add_haze(image_processor, model, image, strength=0.2):
    depth = est_depth(image_processor, model, image, raw_depth=True)
    image_a = np.array(image)
    hazed_a = apply_haze_eq(image_a, depth, scatter_coef=strength)
    return hazed_a


def est_depth(img_proc, model, image, raw_depth=False):
    inputs = img_proc(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # interpolate to original size and visualize the prediction
    post_processed_output = img_proc.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],  # depth anything
        # source_sizes=[(image.height, image.width)],  # zoedepth only
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]
    if raw_depth:
        return predicted_depth.detach().cpu().numpy()

    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth = depth.detach().cpu().numpy() * 255
    return depth


def apply_haze_eq(rgb, depth, scatter_coef=1, atmos_light=230):
    trans = np.exp(-scatter_coef*depth)
    trans = np.stack([trans]*3, axis=-1)

    hazed = rgb*trans + atmos_light*(1-trans)
    hazed = hazed.round().astype("uint8")
    return hazed


if __name__ == "__main__":
    parser = util.get_argparser()
    args = parser.parse_args()
    main(args)
