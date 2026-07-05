"""Generate thumbnails for all HDRI images"""


import logging
import pathlib

import numpy as np
import pandas as pd
import tqdm

import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.safeimport.bpy import bpy


logging.basicConfig(level="INFO")

# --- CLI arguments
parser = kb.ArgumentParser()

# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default="resources/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str,
                    default="gs://kubric-public/assets/GSO/GSO.json")

parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.set_defaults(resolution=128)

FLAGS = parser.parse_args()


# --- Common setups & resources
kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)

hdri_info = pd.read_json("resources/HDRI_haven_cat_tag_df.json")
for asset_id in tqdm.tqdm(hdri_info.asset_id.to_list()):
    # setup
    scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
    renderer = KubricRenderer(scene, scratch_dir, samples_per_pixel=64)
    scene += kb.PerspectiveCamera(name="camera", position=(0, -10, 2),
                                look_at=(0, 0, 0))
    scene.camera.field_of_view = np.radians(90)

    # Background
    bg_hdri = hdri_source.create(asset_id=asset_id)
    scene.metadata["background"] = asset_id
    logging.info("Using background %s", asset_id)
    renderer._set_ambient_light_hdri(bg_hdri.filename, strength=1.0)
    # Dome
    dome = kubasic.create(asset_id="dome", name="dome",
                        static=True, background=True)
    assert isinstance(dome, kb.FileBasedObject)
    scene += dome
    dome_blender = dome.linked_objects[renderer]
    texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node.image = bpy.data.images.load(bg_hdri.filename)

    output_name = pathlib.Path(__file__).stem

    scene.camera.position = (0, -10, 2)
    scene.camera.look_at((0, 0, 0))
    logging.info("Rendering the scene (N) ...")
    frame = renderer.render_still()
    kb.write_png(frame["rgba"], f"output/{output_name}_{asset_id}_n.png")

    scene.camera.position = (-10, 0, 2)
    scene.camera.look_at((0, 0, 0))
    logging.info("Rendering the scene (E) ...")
    frame = renderer.render_still()
    kb.write_png(frame["rgba"], f"output/{output_name}_{asset_id}_e.png")

    scene.camera.position = (0, 10, 2)
    scene.camera.look_at((0, 0, 0))
    logging.info("Rendering the scene (S) ...")
    frame = renderer.render_still()
    kb.write_png(frame["rgba"], f"output/{output_name}_{asset_id}_s.png")

    scene.camera.position = (10, 0, 2)
    scene.camera.look_at((0, 0, 0))
    logging.info("Rendering the scene (W) ...")
    frame = renderer.render_still()
    kb.write_png(frame["rgba"], f"output/{output_name}_{asset_id}_w.png")

    scene.remove(dome)


"""
docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$PWD:/kubric" \
    kubricdockerhub/kubruntu \
    python3 hdri.py
"""
