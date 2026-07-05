import argparse
import datetime as dt
import logging
import os
import pathlib
import pickle
import typing

import pandas as pd
import numpy as np


logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                    filename="oood.log",
                    format="%(asctime)s - %(levelname)s\t - %(message)s - [%(name)s..%(filename)s:%(lineno)d]",
                    datefmt="%Y-%m-%d %H:%M:%S")


DEBUG = os.getenv("DEBUG")
HF_TOKEN = os.getenv('HF_TOKEN')
HF_CACHE = os.getenv('HF_HOME', '~/.cache/huggingface')

o3d_data_dirpath = os.getenv("O3D_DATA_DIRPATH")
if o3d_data_dirpath:
    DEFAULT_DATA_PATH = pathlib.Path(o3d_data_dirpath)
    logging.info("Setting data path from env var O3D_DATA_DIRPATH: %s", DEFAULT_DATA_PATH)
else:
    DEFAULT_DATA_PATH = pathlib.Path(__file__).parent.parent/"data"
    logging.info("Setting data path relatively: %s", DEFAULT_DATA_PATH)

AUG_SUFFIX_REGEX = r"-augmented-[a-z-]+"


class O3DArgumentParser(argparse.ArgumentParser):
    def parse_args(self):
        args = super().parse_args()
        if 'highestc' in args.question_set and not args.image_set.endswith("-mark"):
            raise ValueError("Questions with highest clarity require images"
                             f" to be marked. Got image set: {args.image_set}")
        return args


def timestamp_str(compact=False):
    if compact:
        return dt.datetime.now().strftime("%Y%m%d%H%M%S")
    return dt.datetime.now().replace(microsecond=0).isoformat()


def format_suffix(name, suffix):
    parts = name.rsplit('.', maxsplit=1)
    assert len(parts) == 2
    return f"{parts[0]}{suffix}.{parts[1]}"


def format_subdir(filepath, subdir, sibling=False) -> pathlib.Path:
    filepath = pathlib.Path(filepath)

    dest_dir = filepath.parent
    if sibling:
        dest_dir = dest_dir.parent

    formatted = dest_dir / subdir / filepath.name
    return formatted


def get_argparser():
    parser = O3DArgumentParser(description='Run VQA expts for OOOD')
    parser.add_argument('--model_name', help='Model name', default=None)

    parser.add_argument('--image_set', help='Image set name', default="o3d-real-aug")
    parser.add_argument('--imset_kwargs', help='kwargs passing to get_image_set(), format: k1=v1,k2=v2',
                        default={}, type=_arg_to_dict)

    parser.add_argument('--question_set', help='Question set name', default="min")
    parser.add_argument('--icl', help="In-context learning", action="store_true", default=False)
    parser.add_argument('--cot', help="Chain-of-thoughts", action="store_true", default=False)

    parser.add_argument('--resume', help="Continue from last saved results by timestamp", required=False)
    return parser


def prepare_vlm_run(args):
    if args.resume:
        timestamp = args.resume
        result_records = load_results(args, timestamp)
    else:
        timestamp = timestamp_str(compact=True)
        result_records = []  # list of records
    return timestamp, result_records


def _arg_to_dict(arg_str):
    parts = arg_str.split(",")
    return {p.split("=")[0]: p.split("=")[1] for p in parts}

def save_pickle(data, filename):
    filepath = f"{filename}-{timestamp_str()}.pkl"
    logging.info("Saving pickle to: %s", filepath)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def save_results(args, result_records: typing.List[dict], ts_str=None):
    results_dir = DEFAULT_DATA_PATH / "results" / args.model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    if DEBUG: results_dir = pathlib.Path(".")

    ts_str = ts_str or timestamp_str(compact=True)
    filename = f"{args.image_set}_{args.question_set}_records_{ts_str}.pkl"
    fo_path = results_dir / filename
    logging.info("Saving results to: %s", fo_path)
    with open(fo_path, "wb") as fo:
        pickle.dump(result_records, fo)


def load_results(args, ts_str):
    results_dir = DEFAULT_DATA_PATH / "results" / args.model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    if DEBUG: results_dir = pathlib.Path(".")

    filename = f"{args.image_set}_{args.question_set}_records_{ts_str}.pkl"
    fo_path = results_dir / filename
    logging.info("Loading results from: %s", fo_path)
    with open(fo_path, "rb") as fi:
        result_records = pickle.load(fi)
    return result_records


def load_pickled_to_df(path) -> pd.DataFrame:
    with open(path, "rb") as fi:
        records = pickle.load(fi)
    return pd.DataFrame.from_records(records)


def batches_of(seq, size=1000):
    batch = []
    for n, item in enumerate(seq, start=1):
        batch.append(item)
        if n % size == 0:
            yield batch
            batch = []
    if batch:
        yield batch


def calc_mask_center(mask: np.ndarray):
    """Calculate center of mass of a binary mask."""
    row_indices, col_indices = mask.nonzero()[0], mask.nonzero()[1]
    x_center = col_indices.mean()
    y_center = row_indices.mean()
    return np.array([x_center, y_center])


def df_dataset_get_sample(ds, image_name):
    idx = ds['image_name'].index(image_name)
    return ds[idx]
