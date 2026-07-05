

import logging
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.safeimport.bpy import bpy


logging.basicConfig(level="INFO")

def main():
    scene = kb.Scene(resolution=(256, 256))
    for __ in range(1100):
        cam = kb.PerspectiveCamera(name="camera", position=(3, -1, 4),
                                   look_at=(0, 0, 1))
        scene += cam
        scene.remove(cam)
        logging.info("Removed camera %s", cam.uid)
        # INFO:root:Removed camera camera.1099



if __name__ == "__main__":
    main()
