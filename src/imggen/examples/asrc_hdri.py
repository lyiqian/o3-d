"""Asset (objs) sources & HDRI sources"""


import logging
import pathlib

import numpy as np

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

FLAGS = parser.parse_args()


# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
renderer = KubricRenderer(scene, scratch_dir, samples_per_pixel=64)
kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)

hdri_ids, __ = hdri_source.get_test_split(fraction=0.1)
hdri_id = hdri_ids[0]
bg_hdri = hdri_source.create(asset_id=hdri_id)
scene.metadata["background"] = hdri_id
logging.info("Using background %s", hdri_id)
renderer._set_ambient_light_hdri(bg_hdri.filename, strength=1.0)

# Dome
dome = kubasic.create(asset_id="dome", name="dome",
                      static=True, background=True)
assert isinstance(dome, kb.FileBasedObject)
scene += dome
dome_blender = dome.linked_objects[renderer]
texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
texture_node.image = bpy.data.images.load(bg_hdri.filename)

# Cam
scene += kb.PerspectiveCamera(name="camera", position=(0, -10, 2.02),
                              look_at=(0, 0, 0))

# scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
# scene.camera.position = kb.sample_point_in_half_sphere_shell(
#     inner_radius=7., outer_radius=9., offset=0.1)
# scene.camera.look_at((0, 0, 0))


# Objects
obj_ids, __  = gso.get_test_split(fraction=0.1)
obj_id = obj_ids[1]
obj = gso.create(asset_id=obj_id)
assert isinstance(obj, kb.FileBasedObject)

scale = rng.uniform(0.75, 3.0)
scale = 1  # DEBUG  -> scale of the max dim will be 1 (meter)
obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
obj.metadata["scale"] = scale
logging.info("Object %s; bounds: %s", obj.metadata, obj.bounds)

scene += obj
logging.info("Added %s at %s", obj.asset_id, obj.position)

# --- Rendering
if FLAGS.save_state:
  logging.info("Saving the renderer state to '%s' ",
               output_dir / "scene.blend")
  renderer.save_state(output_dir / "scene.blend")

logging.info("Rendering the scene ...")
frame = renderer.render_still()


# --- save the output as pngs
output_name = pathlib.Path(__file__).stem
kb.write_png(frame["rgba"], f"output/{output_name}.png")
kb.write_palette_png(frame["segmentation"], f"output/{output_name}_segmentation.png")
scale = kb.write_scaled_png(frame["depth"], f"output/{output_name}_depth.png")
logging.info("Depth scale: %s", scale)



"""
docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$PWD:/kubric" \
    kubricdockerhub/kubruntu \
    python3 examples/asrc_hdri.py
"""
