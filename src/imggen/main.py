"""Use kubric (kb) to generate O3-D dataset. https://github.com/google-research/kubric"""
import abc
import datetime as dt
import itertools
import logging
import math
import os
import pathlib
import random

import numpy as np
import pandas as pd
import tqdm
import PIL.Image

import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.safeimport.bpy import bpy

import rawdata
import haze


DEBUG = os.getenv("DEBUG")
DEFAULT_LOGGING_DIRPATH = pathlib.Path("/tmp")
DATA_DIRPATH = pathlib.Path(os.getenv("O3D_DATA_DIRPATH") or "data").resolve()

MAIN_OBJ_NAME = "MainObject"
CAM_OBJ_NAME = "Camera"
GROUND_OBJ_NAME = "Ground"
BG_OBJ_NAME = "Background"
SUNLIGHT_OBJ_NAME = "Sun"
SPECIAL_OBJ_NAMES = {CAM_OBJ_NAME, GROUND_OBJ_NAME, BG_OBJ_NAME, SUNLIGHT_OBJ_NAME}

BASIC_ENV_TYPE = 'basic'
HDRI_ENV_TYPE = 'hdri'

TARGET_SEG_ID = 1
DISTRACTOR_SEG_ID = 2
OTHER_SEG_ID = 0

SELECTED_ENVIRONMENTS = [
    (BASIC_ENV_TYPE, None),
    (HDRI_ENV_TYPE, "mud_road"),
    (HDRI_ENV_TYPE, "umhlanga_sunrise"),
    (HDRI_ENV_TYPE, "aristea_wreck"),
    (HDRI_ENV_TYPE, "lenong_3"),
    (HDRI_ENV_TYPE, "evening_road_01"),
    (HDRI_ENV_TYPE, "waterbuck_trail"),
    (HDRI_ENV_TYPE, "abandoned_hall_01"),  # low horizon contrast
    (HDRI_ENV_TYPE, "aerodynamics_workshop"),  # low ground texture
    (HDRI_ENV_TYPE, "castle_zavelstein_cellar"),  # light ground dark bg
    (HDRI_ENV_TYPE, "dresden_station_night"),  # night, semi-outdoor
    (HDRI_ENV_TYPE, "parking_garage"),  # common scene
    (HDRI_ENV_TYPE, "royal_esplanade"),  # complex bg
]


SELECTED_OBJECTS = [
    ("kubasic", "cube"),
    ("kubasic", "sphere"),
    ("kubasic", "sponge"),  # cube with holes
    ("kubasic", "torus_knot"),
    ("kubasic", "suzanne"),  # cartoon char
    ("kubasic", "spot"),  # cartoon cow
    ("gso", "Threshold_Porcelain_Teapot_White"),  # teapot in GSO
    ("gso", "W_Lou_z0dkC78niiZ"),  # boot
    ("gso", "Ecoforms_Plant_Container_Quadra_Sand_QP6"),  # plant pot
    ("gso", "Dog"), # toy
    ("gso", "Nickelodeon_Teenage_Mutant_Ninja_Turtles_Michelangelo"), # complex shape
    ("gso", "Retail_Leadership_Summit_tQFCizMt6g0"), # hat
    ("gso", "Digital_Camo_Double_Decker_Lunch_Bag"),
    ("gso", "Curver_Storage_Bin_Black_Small"),
    ("gso", "Threshold_Porcelain_Pitcher_White"),  # pitcher
    ("gso", "HyperX_Cloud_II_Headset_Red"),  # box with texture
    ("gso", "Great_Dinos_Triceratops_Toy"),  # android
    ("gso", "Travel_Mate_P_series_Notebook"),  # laptop
    ("gso", "Olive_Kids_Birdie_Sidekick_Backpack"),  # backpack
    ("gso", "Mens_Bahama_in_Black_b4ADzYywRHl"),  # shoe
    ("gso", "Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White"),  # mug
    ("gso", "Nintendo_Mario_Action_Figure"),  # mario
    ("gso", "BABY_CAR"),  # plastic toy
    ("gso", "BIRD_RATTLE"),  # fluffy toy
    ("gso", "Black_Decker_Stainless_Steel_Toaster_4_Slice"),
    ("gso", "TriStar_Products_PPC_Power_Pressure_Cooker_XL_in_Black"),  # pressure cooker
    ("gso", "Pennington_Electric_Pot_Cabana_4"),  # small pot
    ("gso", "Crosley_Alarm_Clock_Vintage_Metal"),  # metal clock
    ("gso", "Threshold_Bead_Cereal_Bowl_White"),  # bowl
    ("gso", "Ortho_Forward_Facing_CkAW6rL25xH"),  # helmet
    ("gso", "Toys_R_Us_Treat_Dispenser_Smart_Puzzle_Foobler"),  # sphere dispenser
    ("gso", "CoQ10_BjTLbuRVt1t"),  # medicine bottle
    ("gso", "Granimals_20_Wooden_ABC_Blocks_Wagon_85VdSftGsLi"),  # wooden block
    ("gso", "ASICS_GELDirt_Dog_4_SunFlameBlack"),  # colorful shoe
    ("gso", "Air_Hogs_Wind_Flyers_Set_Airplane_Red"),  # red plane toy
    ("gso", "Dino_5"),  # grey dinosaur
    ("gso", "Paint_Maker"),  # colorful box
]

FAMILIAR_SIZE_OBJECT_PAIRS = [  # large vs small
    [("gso", "Organic_Whey_Protein_Unflavored"), ("gso", "CoQ10_BjTLbuRVt1t")],
    [("gso", "Travel_Mate_P_series_Notebook"), ("gso", "BlackBlack_Nintendo_3DSXL")],
    [("gso", "Threshold_Porcelain_Pitcher_White"), ("gso", "Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White")],
    [("gso", "Remington_TStudio_Hair_Dryer"), ("gso", "Razer_Abyssus_Ambidextrous_Gaming_Mouse")],
    [("gso", "TriStar_Products_PPC_Power_Pressure_Cooker_XL_in_Black"), ("gso", "Threshold_Porcelain_Teapot_White")],
]

OBJ_TO_ROTATE = {
    "suzanne",
    "spot",
    "BABY_CAR",
    "BIRD_RATTLE",
}

ODD_NONE = 'none'
ODD_DEPTH_FAR = 'far'
ODD_DEPTH_NEAR = 'near'

CUE_SHADOW = 'LS'  # Light & Shadow
CUE_HEIGHT_IN_PLANE = 'HP'
CUE_HEIGHT_IN_PLANE_CTRL = 'HPC'
CUE_OCCLUSION = 'OC'
CUE_TEXTURE_GRADIENT = 'TG'
CUE_RELATIVE_SIZE = 'RS'
CUE_FOCUS = 'FO'
CUE_SATURATION = 'SA'

CUE_FAMILIAR_SIZE = 'FS'
# CUE_LINEAR_PERSPECTIVE = 'LP'  # using args.elim_lp instead

CUE_CHOICES = [
    CUE_TEXTURE_GRADIENT,
    CUE_OCCLUSION,
    CUE_HEIGHT_IN_PLANE,
    CUE_SHADOW,
    CUE_FOCUS,
    CUE_RELATIVE_SIZE,
    CUE_SATURATION,
    CUE_FAMILIAR_SIZE,
]

DEFAULT_CAM_DEPTH = -20
DEFAULT_CAM_HEIGHT = 0
DEFAULT_HDRI_DOME_SIZE = 40
ODD_OBJ_SCALER = 1.2  # SA cue will work better with greater scaler (larger D diff)
ODD_SCALER_MIN = 1.1
ODD_SCALER_MAX = 2.0
MAX_CUE_STRENGTH = 3.0
# DEFAULT_CUE_STRENGTH_VALUES = np.logspace(-1, np.log10(MAX_CUE_STRENGTH), num=10)
DEFAULT_CUE_STRENGTH_VALUES = [1, 2]

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                    format="%(asctime)s - %(levelname)s\t - %(message)s - [%(name)s..%(filename)s:%(lineno)d]",
                    datefmt="%Y-%m-%d %H:%M:%S")

lgr = logging



class KbO3dScene(rawdata.O3dSceneAbc):
    def __init__(self, *args, **kwargs):
        self.kb_scene = kwargs.pop("kb_scene")  # saving native kb.Scene() from `kubric` package
        super().__init__(*args, **kwargs)

    @classmethod
    def from_blend(cls, path):
        pass  # todo

    def to_blend(self, path):
        pass  # todo


class KbBaseSceneBuilder(rawdata.BaseSceneBuilderAbc):
    """Build a minimum scene with, say 5, identical objects in a row."""
    MAX_NAME_LEN = 54  # otherwise non-friendly for bpy.context.scene.objects.keys

    def build(self, obj_src: kb.AssetSource, kb_scene: kb.Scene) -> KbO3dScene:
        scene, target = self._init_scene_with_target(obj_src, kb_scene)

        scene = self._add_distractors(obj_src, scene, target)

        self._add_camera(scene)

        return scene

    def add_oddity(
        self, scene: KbO3dScene, position: str,
        obj_src: kb.AssetSource=None,
    ) -> KbO3dScene:
        if position == ODD_NONE:
            oddity_ops = [rawdata.SceneNoOp()]

        elif position == ODD_DEPTH_NEAR:
            oddity_ops = [  # move distractors farther
                DollyOperator("distractors", obj_src, coef=self.odd_scaler, distractor_idx=i)
                for i in range(self.n_distractors)
            ]
            oddity_ops.append(
                ObjectPlacementOperator("distractors", obj_src))

        else:  # far
            oddity_ops = [  # move target farther
                DollyOperator("target", obj_src, coef=self.odd_scaler),
                ObjectPlacementOperator("target", obj_src)
            ]

        lgr.info("Adding oddity [%s] with ops: %s", position, oddity_ops)
        for oddity_op in oddity_ops:
            scene = oddity_op.change(scene)

        return scene

    def _init_scene_with_target(self, obj_src, kb_scene):
        assert len(self.target_name) <= 54, "target_name too long"
        scene = KbO3dScene(self.target_name, self.n_distractors, kb_scene=kb_scene)

        # Some objects need rotation (`distractors` unexpectedly inherit `target` rotation)
        euler = (0, 0, np.pi) if self.target_name in OBJ_TO_ROTATE else None
        target = obj_src.create(asset_id=self.target_name, name=self.target_name,
                                segmentation_id=TARGET_SEG_ID, euler=euler)

        scene.kb_scene += target
        scene.target = target
        obj_place_op = ObjectPlacementOperator("target", obj_src)
        scene = obj_place_op.change(scene)
        target = scene.target
        lgr.info("Added target: %s @ %s", target.uid, target.position)
        return scene, target

    def _add_distractors(self, obj_src, scene, target):
        scene.distractors = []
        for i in range(self.n_distractors):
            # unit offset is simply: obj width * odd_scaler;
            # changing this will break the calc for cam position for occlusion (OC)
            multiplier =  self.odd_scaler * (i//2+1) * (-1)**i  # alternate positions
            x_offset = _obj_dims(target)[0] * multiplier
            distractor = obj_src.create(
                asset_id=target.asset_id,
                name=target.name+"_copy",
                position=(x_offset, 0, 0),
                segmentation_id=DISTRACTOR_SEG_ID)

            scene.kb_scene += distractor
            scene.distractors.append(distractor)
            lgr.info("Added distractor: %s (%sX) @ %s",
                     distractor.uid, distractor.scale, distractor.position)

        obj_place_op = ObjectPlacementOperator("distractors", obj_src)
        scene = obj_place_op.change(scene)
        return scene

    def _add_camera(self, scene):
        camera = kb.PerspectiveCamera(
            name=CAM_OBJ_NAME,
            position=(0, self._get_cam_depth(scene), DEFAULT_CAM_HEIGHT),
            look_at=(0, 0, 0))
        scene.kb_scene += camera
        scene.camera = camera
        lgr.info("Added camera: %s @ %s", camera.uid, camera.position)

    def _get_cam_depth(self, scene: KbO3dScene):
        scaler = _obj_dims(scene.target)[0]
        return DEFAULT_CAM_DEPTH * scaler


class FamiliarSizeKbBaseSceneBuilder(KbBaseSceneBuilder):

    def _add_distractors(self, obj_src, scene, target):
        scene.distractors = []
        for i in range(self.n_distractors):
            # unit offset is simply: obj width * odd_scaler;
            # changing this will break the calc for cam position for occlusion (OC)
            multiplier =  1.5 * (i//2+1) * (-1)**i  # alternate positions
            x_offset = _obj_dims(target)[0] * multiplier
            distractor = obj_src.create(
                asset_id=self.distractor_name,
                name=self.distractor_name,
                position=(x_offset, 0, 0),
                segmentation_id=DISTRACTOR_SEG_ID)

            scene.kb_scene += distractor
            scene.distractors.append(distractor)
            lgr.info("Added distractor: %s (%sX) @ %s",
                     distractor.uid, distractor.scale, distractor.position)

        obj_place_op = ObjectPlacementOperator("distractors", obj_src)
        scene = obj_place_op.change(scene)
        return scene

    def add_oddity(
        self, scene: KbO3dScene, position: str,
        obj_src: kb.AssetSource=None,
    ) -> KbO3dScene:
        if position == ODD_NONE:
            raise ValueError("Familiar Size (FS) does not support none oddity.")

        elif position == ODD_DEPTH_NEAR:
            coef = _obj_dims(scene.distractors[0])[2] / _obj_dims(scene.target)[2]  # height ratio
            oddity_ops = [
                DollyOperator("distractors", obj_src, coef=coef, distractor_idx=i, pos_only=True)
                for i in range(self.n_distractors)
            ]
            oddity_ops.append(
                ObjectPlacementOperator("distractors", obj_src))

        else:  # far
            coef =  _obj_dims(scene.target)[2] / _obj_dims(scene.distractors[0])[2]  # height ratio
            oddity_ops = [
                DollyOperator("target", obj_src, coef=coef, pos_only=True),
                ObjectPlacementOperator("target", obj_src)
            ]

        lgr.info("Adding oddity [%s] with ops: %s", position, oddity_ops)
        for oddity_op in oddity_ops:
            scene = oddity_op.change(scene)

        return scene



class KbRenderer(rawdata.RendererAbc):
    core: KubricRenderer

    def __init__(self, core: KubricRenderer):
        self.core = core

    def render(self, scene: KbO3dScene) -> rawdata.RenderResult:
        frame = self.core.render_still()

        result = rawdata.RenderResult(
            image=frame["rgba"],
            segmt=frame["segmentation"],
            depth=frame["depth"])
        return result


class EnvironmentBuilderAbc(abc.ABC):
    """Build env like ground, background, and lighting."""

    @abc.abstractmethod
    def build(self, **kwargs) -> tuple[rawdata.BackgroundT, rawdata.GroundT, rawdata.LightT]:
        pass


class KbMinEnvBuilder(EnvironmentBuilderAbc):
    def __init__(self, kb_scene):
        self.kb_scene = kb_scene

    def build(self, **kwargs):
        ground = kb.Cube(
            name="ground",
            scale=(DEFAULT_HDRI_DOME_SIZE, DEFAULT_HDRI_DOME_SIZE, 0.1),
            position=(0, 0, -0.1))
        bg = kb.Cube(
            name="background",
            scale=(2*DEFAULT_HDRI_DOME_SIZE, 0.1, DEFAULT_HDRI_DOME_SIZE),
            position=(0, DEFAULT_HDRI_DOME_SIZE, DEFAULT_HDRI_DOME_SIZE))
        underground = kb.Cube(
            name="underground",
            scale=(2*DEFAULT_HDRI_DOME_SIZE, 0.1, DEFAULT_HDRI_DOME_SIZE),
            position=(0, 0, -DEFAULT_HDRI_DOME_SIZE-0.001))
        light = kb.DirectionalLight(
            name="sun", position=(0, -5, 40), look_at=(0, 0, 0), intensity=1)

        self.kb_scene += ground
        self.kb_scene += bg
        self.kb_scene += underground
        self.kb_scene += light

        self.textureless_ground = kwargs.get('textureless_ground')  # remove LP cue
        if self.textureless_ground:
            pass
        else:
            ground_tex_op = GroundTextureOperator(
                frequency=5,
                colors=[kb.core.color.Color.from_name("gray"),
                        kb.core.color.Color.from_name("white")])
            ground_tex_op.add_procedure_texture(ground, scale=1.0)

        bg.material = kb.PrincipledBSDFMaterial(color=kb.Color(0.4, 0.5, 0.8))
        underground.material = kb.PrincipledBSDFMaterial(color=kb.Color(0, 0, 0), roughness=0)

        # self.kb_scene.add(kb.assets.utils.get_clevr_lights())
        self.kb_scene.ambient_illumination = kb.Color(0.1, 0.1, 0.1)

        return bg, ground, light


class KbHdriEnvBuilder(EnvironmentBuilderAbc):
    def __init__(self, kb_scene, kb_renderer, hdri_source, dome=None):
        self.kb_scene = kb_scene
        self.kb_renderer = kb_renderer
        self.hdri_source = hdri_source
        self.dome = dome

    def build(self, env_id=None):
        if env_id is None:
            hdri_ids, __ = self.hdri_source.get_test_split(fraction=0.1)
            env_id = hdri_ids[0]

        lgr.info("Using HDRI background: %s", env_id)
        bg_hdri = self.hdri_source.create(asset_id=env_id)
        bg_hdri.segmentation_id = OTHER_SEG_ID
        self.kb_renderer._set_ambient_light_hdri(bg_hdri.filename, strength=1.0)

        # self.kb_scene.metadata["background"] = env_id

        assert isinstance(self.dome, kb.FileBasedObject)
        self.kb_scene += self.dome
        dome_blender = self.dome.linked_objects[self.kb_renderer]
        texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
        texture_node.image = bpy.data.images.load(bg_hdri.filename)

        underground = kb.Cube(
            name="underground",
            scale=(2*DEFAULT_HDRI_DOME_SIZE, 0.1, DEFAULT_HDRI_DOME_SIZE),
            position=(0, 0, -DEFAULT_HDRI_DOME_SIZE-0.001))  # matching dome position
        underground.material = kb.PrincipledBSDFMaterial(color=kb.Color(0, 0, 0), roughness=0)
        self.kb_scene += underground

        return self.dome, None, None


class KbRenderable(rawdata.Renderable):
    scene: KbO3dScene
    renderer: KbRenderer
    environment_builder: EnvironmentBuilderAbc

    def __init__(self, scene, renderer, env_builder):
        super().__init__(scene, renderer)
        self.environment_builder = env_builder


    def render(self) -> rawdata.RenderResult:
        render_result = super().render()
        render_result.targ_labels = self._adjust_seg_idxs(
            render_result.segmt, [TARGET_SEG_ID])
        render_result.dist_labels = self._adjust_seg_idxs(
            render_result.segmt, [DISTRACTOR_SEG_ID])

        render_result.tsegmt = self._adjust_seg_idxs(  # target raw seg_id
            render_result.segmt,
            [i for i, a in enumerate(self.scene.kb_scene.assets, start=1) if a in {self.scene.target}],
            raw=True)
        render_result.dsegmt = self._adjust_seg_idxs(  # distractor raw seg_ids
            render_result.segmt,
            [i for i, a in enumerate(self.scene.kb_scene.assets, start=1) if a in set(self.scene.distractors)],
            raw=True)
        return render_result

    def setup_environment(self, **kwargs):
        bg, gd, light = self.environment_builder.build(**kwargs)

        self.scene.background = bg
        self.scene.ground = gd
        self.scene.light = light

    def _adjust_seg_idxs(self, seg, seg_ids_to_keep, raw=False):
        if raw:
            seg_ = seg
        else: # make manually assigned seg_id working
            seg_ = kb.adjust_segmentation_idxs(seg, self.scene.kb_scene.assets, [])

        adjusted_seg = np.zeros_like(seg_)
        for seg_id in seg_ids_to_keep:
            adjusted_seg[seg_ == seg_id] = seg_id
        return adjusted_seg


class KbCueControler(rawdata.CueControlerAbc):
    def add_cue(self, renderable: KbRenderable, strength=1.0) -> KbRenderable:
        for scene_op in self.scene_operators:
            scene_op.change(renderable.scene, scale=strength)
        return renderable

    def remove_cue(self, renderable: KbRenderable, strength=1.0) -> KbRenderable:
        for scene_op in self.scene_operators[::-1]:
            scene_op.unchange(renderable.scene, scale=strength)
        return renderable


class KbRelativeSizeCueControler(KbCueControler):
    """For adjusting cue strengths for RS cue depending on oddity position."""
    def __init__(self, scene_operators, odd_scaler, odd_position=ODD_NONE, strength_factor=1.0):
        super().__init__(scene_operators)
        self.odd_scaler = odd_scaler
        self.odd_position = odd_position
        self.strength_factor = strength_factor  # for normalization

    def add_cue(self, renderable: KbRenderable, strength=1.0) -> KbRenderable:
        # ensuring 2 conditions for strength: 0 = no change, 1 = natural size
        # only works for "target" not "distractors"
        scaler_offset = self.odd_scaler - 1
        base_scaler = 1 + scaler_offset * (strength**self.strength_factor)

        if self.odd_position == ODD_DEPTH_FAR:  # make target smaller
            strength_ = 1 / base_scaler
        else:
            strength_ = base_scaler

        return super().add_cue(renderable, strength=strength_)

    def remove_cue(self, renderable: KbRenderable, strength=1.0) -> KbRenderable:
        # strength does not affect cue removal for scale ops
        return super().remove_cue(renderable, strength=strength)


class CameraPositionOperator(rawdata.ISceneOperator):
    def __init__(self, axis, unit_dist=1, base_dist=0, relative=False):
        self.axis = axis
        self.unit_dist = unit_dist
        self.base_dist = base_dist
        self.relative = relative

    def __str__(self):
        return f"ax={self.axis}; unit={self.unit_dist}; base={self.base_dist}; rel={self.relative}"

    def change(self, scene: KbO3dScene, scale=1.0):
        lgr.info("Changing camera [%s]X w/ %s", scale, self)
        self.orig_position = scene.camera.position[self.axis]

        self._move_cam(
            scene, scene.camera,
            self.axis, self.base_dist + scale*self.unit_dist,
            self.relative)
        return scene

    def unchange(self, scene: KbO3dScene, scale=1.0):
        if self.relative:
            self._move_cam(
                scene, scene.camera,
                self.axis, -self.base_dist - scale*self.unit_dist,
                relative=True)
        else:
            self._move_cam(
                scene, scene.camera, self.axis, self.orig_position)
        return scene

    @staticmethod
    def _move_cam(
        scene: KbO3dScene,
        cam: rawdata.CameraT,
        axis, dist, relative,
    ):
        """Move by recreating the camera. (no panning to keep relative distances)"""

        new_location = list(cam.position)

        if relative:
            new_location[axis] += dist
        else:
            new_location[axis] = dist
        look_at_x = new_location[0]

        scene.kb_scene.remove(cam)
        moved_cam = kb.PerspectiveCamera(
            name=cam.name,
            position=tuple(new_location),
            look_at=(look_at_x, 0, 0))
        scene.kb_scene += moved_cam
        scene.camera = moved_cam

        return moved_cam


class CameraDOFOperator(rawdata.ISceneOperator):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def change(self, scene: KbO3dScene, scale=1.0):
        bpy_cam = bpy.data.objects[scene.camera.uid]

        bpy_cam.data.dof.use_dof = self.enabled

        focus_dist = scene.target.position[1] - scene.camera.position[1]
        bpy_cam.data.dof.focus_distance = focus_dist
        bpy_cam.data.dof.aperture_fstop = 2.8 / scale
        lgr.info("Set camera focus distance: %.2f; aperture: %.2f",
                 bpy_cam.data.dof.focus_distance, bpy_cam.data.dof.aperture_fstop)

        return scene

    def unchange(self, scene: KbO3dScene, scale=1.0):
        bpy_cam = bpy.data.objects[scene.camera.uid]
        bpy_cam.data.dof.use_dof = not bpy_cam.data.dof.use_dof
        return scene


class LightOperator(rawdata.ISceneOperator):
    def __init__(self, position, unit_intensity=0.5,
                 look_at=(0, 0, 0)):  # todo minimize effect on bg
        self.position = position
        self.look_at = look_at
        self.unit_intensity = unit_intensity

    def change(self, scene: KbO3dScene, scale=1.0):
        lgr.info("Adding a sun light for LS cue @ %s", self.position)
        cue_light = kb.DirectionalLight(
            name="cue_light", position=self.position, look_at=self.look_at,
            intensity=scale*self.unit_intensity,
        )
        scene.kb_scene += cue_light
        self.cue_light = cue_light
        return scene

    def unchange(self, scene: KbO3dScene, scale=1.0):
        scene.kb_scene.remove(self.cue_light)
        return scene


class ObjectPositionOperator(rawdata.ISceneOperator):
    def __init__(self, scene_attr, axis, unit_dist=1, relative=False, obj_src=None):
        assert scene_attr in {"target", "distractors"}, f"Invalid attr: {scene_attr}"
        self.scene_attr = scene_attr
        self.axis = axis
        self.unit_dist = unit_dist
        self.relative = relative
        self.obj_src = obj_src

    def __str__(self):
        return f"{self.scene_attr} / axis={self.axis}; unit={self.unit_dist}; rel={self.relative}"

    def change(self, scene: KbO3dScene, scale=1.0):
        lgr.info("Positioning: [%s]X w/ %s", scale, self)
        op_targ = getattr(scene, self.scene_attr)

        if self.scene_attr == "distractors":
            self.orig_pos = []
            moved_distractors = []
            for o in op_targ:  # only useful for same relative translation for distractors
                self.orig_pos.append(o.position[self.axis])
                moved_o = self._move_obj(scene, o, scale)
                moved_distractors.append(moved_o)
            scene.distractors = moved_distractors  # could also use setattr

        else:
            self.orig_pos = op_targ.position[self.axis]
            moved_targ = self._move_obj(scene, op_targ, scale)
            scene.target = moved_targ  # could also use setattr

        return scene

    def unchange(self, scene: KbO3dScene, scale=1.0):
        lgr.info("Undoing position: [%s]X w/ %s", scale, self)
        op_targ = getattr(scene, self.scene_attr)

        if self.scene_attr == "distractors":
            unmoved_distractors = []
            for o, orig_pos in zip(op_targ, self.orig_pos):
                unmoved_o = self._unmove_obj(scene, o, orig_pos)
                unmoved_distractors.append(unmoved_o)
            scene.distractors = unmoved_distractors

        else:
            unmoved_targ = self._unmove_obj(scene, op_targ, self.orig_pos)
            scene.target = unmoved_targ

        return scene

    def _move_obj(self, scene: KbO3dScene, obj, scale):
        moved_obj = self.move_obj(
            scene.kb_scene, self.obj_src,
            obj, self.axis, scale*self.unit_dist, relative=self.relative)
        return moved_obj

    def _unmove_obj(self, scene: KbO3dScene, obj, orig_pos):
        unmoved_obj = self.move_obj(
            scene.kb_scene, self.obj_src,
            obj, self.axis, orig_pos)
        return unmoved_obj

    @staticmethod
    def move_obj(scene: kb.Scene, obj_src: kb.AssetSource,
                 obj, axis, distance, relative=False):
        new_location = list(obj.position)
        if relative:
            new_location[axis] += distance
        else:
            new_location[axis] = distance

        scene.remove(obj)
        moved_obj = obj_src.create(
            asset_id=_name_to_id(obj.name), name=obj.name,
            position=new_location, scale=obj.scale,
            segmentation_id=obj.segmentation_id)
        lgr.debug("Moved %s to %s", moved_obj.uid, moved_obj.position)
        scene += moved_obj
        return moved_obj


class ObjectPlacementOperator(rawdata.ISceneOperator):
    """Place object(s) on ground based on their height(s).

    Needed whenever obj is created, scaled or moved (due to non-mutable nature)."""
    def __init__(self, scene_attr, obj_src):
        self.scene_attr = scene_attr
        self.obj_src = obj_src

    def change(self, scene: KbO3dScene, scale=1.0):
        op_targ = getattr(scene, self.scene_attr)
        if not op_targ:
            return scene

        if self.scene_attr == "distractors":
            op_targ = op_targ[0]  # assuming same heights

        axis_aligned_lowest_pos = op_targ.aabbox[0][2]
        self.obj_positon_op = ObjectPositionOperator(
            self.scene_attr,
            axis=2, unit_dist=-axis_aligned_lowest_pos, relative=True,
            obj_src=self.obj_src
        )

        lgr.info("Placing obj (%s) on ground", op_targ.uid)
        scene = self.obj_positon_op.change(scene)
        return scene

    def unchange(self, scene, scale=1.0):
        scene = self.obj_positon_op.unchange(scene)
        return scene


class ScaleOperator(rawdata.ISceneOperator):
    def __init__(self, scene_attr, unit_scale=1, absolute=False):
        self.scene_attr = scene_attr
        self.unit_scale = unit_scale
        self.absolute = absolute

    def __str__(self):
        return f"{self.scene_attr} / unit={self.unit_scale}; abs={self.absolute}"

    def change(self, scene: KbO3dScene, scale=1.0):
        lgr.info("Scaling: [%s]X w/ %s", scale, self)
        op_targ = getattr(scene, self.scene_attr)

        if isinstance(op_targ, list):
            self.orig_scale = []
            for o in op_targ:
                self.orig_scale.append(o.scale)
                self._change_scale(o, scale)  # in-place

        else:
            self.orig_scale = op_targ.scale
            self._change_scale(op_targ, scale)  # in-place

        return scene

    def unchange(self, scene: KbO3dScene, scale=1.0):
        lgr.info("Undoing scale: [%s]X w/ %s", scale, self)
        op_targ = getattr(scene, self.scene_attr)

        if isinstance(op_targ, list):
            for o, orig_scale in zip(op_targ, self.orig_scale):
                self._change_scale(o, orig_scale)  # in-place

        else:
            self._change_scale(op_targ, self.orig_scale)  # in-place

        return scene


    def _change_scale(self, obj, scale):
        if self.absolute:  # make the greatest dim equal to `scale`
            desired_scale = scale*self.unit_scale / np.max(obj.bounds[1] - obj.bounds[0])
        else:
            desired_scale = scale*self.unit_scale

        lgr.info("Changing %s scale to: %s", obj.uid, desired_scale)
        obj.scale = desired_scale

        return obj


class DollyOperator(rawdata.ISceneOperator):
    def __init__(self, scene_attr: str, obj_src: kb.AssetSource,
                 coef: float, distractor_idx: int=None, pos_only=False):
        if scene_attr == "distractors":
            assert distractor_idx is not None, "distractor_idx must be set for distractors"

        self.scene_attr = scene_attr  # both supported, although mainly for distractors
        self.obj_src = obj_src
        self.coef = coef
        self.distractor_idx = distractor_idx
        self.pos_only = pos_only  # for FS cue only

    def change(self, scene: KbO3dScene, scale=1.0) -> KbO3dScene:
        op_targ = getattr(scene, self.scene_attr)
        if self.distractor_idx is not None:
            op_targ = op_targ[self.distractor_idx]
        lgr.info("Dolly transforming %s [%s]X", op_targ.uid, scale)

        # scale
        op_targ.scale = scale if self.pos_only else scale*self.coef

        # depth offset
        curr_depth = op_targ.position[1] - scene.camera.position[1]
        desired_depth = curr_depth * scale*self.coef
        depth_offset = desired_depth - curr_depth
        lgr.debug("Dolly offsetting %s depth by %.2f", op_targ.uid, depth_offset)
        moved_obj = ObjectPositionOperator.move_obj(
            scene.kb_scene, self.obj_src,
            op_targ, axis=1, distance=depth_offset, relative=True)
        if self.distractor_idx is not None:
            getattr(scene, self.scene_attr)[self.distractor_idx] = moved_obj
        else:
            setattr(scene, self.scene_attr, moved_obj)

        # (re-getting op_targ after above updates)
        op_targ = getattr(scene, self.scene_attr)
        if self.distractor_idx is not None:
            op_targ = op_targ[self.distractor_idx]

        # horizontal offset
        curr_rel_x = op_targ.position[0] - scene.camera.position[0]
        desired_rel_x = curr_rel_x * scale*self.coef
        x_offset = desired_rel_x - curr_rel_x
        lgr.debug("Offsetting %s horizontal translation by %.2f", op_targ.uid, x_offset)
        moved_obj = ObjectPositionOperator.move_obj(
            scene.kb_scene, self.obj_src,
            op_targ, axis=0, distance=x_offset, relative=True)
        if self.distractor_idx is not None:
            getattr(scene, self.scene_attr)[self.distractor_idx] = moved_obj
        else:
            setattr(scene, self.scene_attr, moved_obj)

        return scene

    def unchange(self, scene: KbO3dScene, scale=1.0):
        raise NotImplementedError("Dolly operator does not support `unchange`")


class TargetObjTextureOperator(rawdata.ISceneOperator):
    def __init__(self, frequency: float,
                 colors=[kb.core.color.Color.from_name("black"),
                         kb.core.color.Color.from_name("white")],
                 odd_position=None,
                ):
        self.freq = frequency
        self.colors = colors
        self.odd_position = odd_position

    def change(self, scene: KbO3dScene, scale=1.0):  # in-place
        # only increase the farther obj texture density
        if self.odd_position == ODD_DEPTH_FAR:  # target is farther
            scale += 1.0
            assert scale >= 1.0
        else:
            scale = 1.0

        self.add_procedure_texture(scene.target, scale)
        return scene

    def add_procedure_texture(self, op_targ, scale):
        """Note: For `scale`, vary args.odd_scaler manually"""
        lgr.info("Adding texture to %s [freq=%s]", op_targ.uid, self.freq*scale)
        op_targ.material = kb.PrincipledBSDFMaterial(name="material")
        op_targ.material.metallic = 0.0
        op_targ.material.roughness = 1.0

        lgr.debug("obj uid: %s", op_targ.uid)
        lgr.debug("scene obj keys: %s", bpy.context.scene.objects.keys())
        mat = bpy.context.scene.objects[op_targ.uid].active_material
        tree = mat.node_tree

        mat_node = tree.nodes["Principled BSDF"]
        ramp_node = tree.nodes.new(type="ShaderNodeValToRGB")
        tex_node = tree.nodes.new(type="ShaderNodeTexNoise")
        scaling_node = tree.nodes.new(type="ShaderNodeMapping")
        location_node = tree.nodes.new(type="ShaderNodeMapping")
        vector_node = tree.nodes.new(type="ShaderNodeNewGeometry")

        tree.links.new(vector_node.outputs["Position"], location_node.inputs["Vector"])
        tree.links.new(location_node.outputs["Vector"], scaling_node.inputs["Vector"])
        tree.links.new(scaling_node.outputs["Vector"], tex_node.inputs["Vector"])
        tree.links.new(tex_node.outputs["Fac"], ramp_node.inputs["Fac"])
        tree.links.new(ramp_node.outputs["Color"], mat_node.inputs["Base Color"])

        location_node.inputs["Location"].default_value = self._get_location_inputs()
        scaling_node.inputs["Scale"].default_value = self._get_scaling_inputs(scale)
        tex_node.inputs["Roughness"].default_value = 0.0
        tex_node.inputs["Detail"].default_value = 0.0

        for x in np.linspace(0.0, 1.0, len(self.colors)+2)[1:-1]:
            ramp_node.color_ramp.elements.new(x)

        for i, element in enumerate(ramp_node.color_ramp.elements):
            element.color = self._get_ramp_color(i)

    def unchange(self, scene: KbO3dScene, scale=1.0):
        return scene  # no need to unchange; just use `change` to overwrite

    def _get_location_inputs(self):
        return (
            0,
            0,
            np.random.uniform(),
        )

    def _get_scaling_inputs(self, scale):
        return (
            0,
            0,  # no change in X & Y -> horizontal stripes
            self.freq * scale,
        )

    def _get_ramp_color(self, i):
        color = self.colors[i%len(self.colors)]
        return color


class DistractorObjTextureOperator(TargetObjTextureOperator):
    def change(self, scene: KbO3dScene, scale=1.0):  # in-place
        # only increase the farther obj texture density
        if self.odd_position == ODD_DEPTH_NEAR:  # distractors are farther
            scale += 1.0
            assert scale >= 1.0
        else:
            scale = 1.0

        for op_targ in scene.distractors:
            self.add_procedure_texture(op_targ, scale)
        return scene


class GroundTextureOperator(TargetObjTextureOperator):
    def change(self, scene: KbO3dScene, scale=1.0):
        self.add_procedure_texture(scene.ground, scale)
        return scene

    def unchange(self, scene, scale=1.0):
        raise NotImplemented

    def _get_location_inputs(self):
        return (
            0,
            0,
            np.random.uniform(),
        )

    def _get_scaling_inputs(self, scale):
        return (
            self.freq * scale,
            self.freq * scale,
            self.freq * scale,
        )


def main(args):
    validate_args(args)
    if args.elim_lp and CUE_HEIGHT_IN_PLANE not in args.cue:
        lgr.info("LP only works when HP cue is present; skipping cues: %s", args.cue)
        return

    records = []
    run_name = format_run_name()

    for penv, pobj in tqdm.tqdm(list(_base_scene_params(args))):
        args.odd_scaler = random.uniform(ODD_SCALER_MIN, ODD_SCALER_MAX)
        lgr.info("Using randomized obj scaler ~= %.2f", args.odd_scaler)

        _scene, __, __, scratch_dir = kb.setup(args)
        kubasic = kb.AssetSource.from_manifest(args.kubasic_assets)
        gso = kb.AssetSource.from_manifest(args.gso_assets)

        if CUE_FAMILIAR_SIZE in args.cue:
            obj_src_name, target_name = pobj[0]
            obj_src_name, small_target_name = pobj[1]
        else:
            obj_src_name, target_name = pobj
            small_target_name = None
        obj_src = kubasic if obj_src_name == "kubasic" else gso

        base_scene = setup_base_scene(args, _scene, obj_src, target_name, small_targ_name=small_target_name)
        renderer = KbRenderer(core=KubricRenderer(_scene, scratch_dir, samples_per_pixel=64))
        renderable = setup_renderable(args, penv, base_scene, renderer, kubasic)

        if not args.cue:  # render base scene, as negative example
            render_result = renderable.render()
            fname_stem = format_filename_stem(args, pobj)
            dscale = write_results(render_result, fname_stem, run_name)
            records.append(_record(args, penv, pobj, fname_stem, dscale))
            continue

        cue_controlers = []
        for pcue in args.cue:
            cue_ctrl = get_cue_controler(pcue, args.odd_scaler,
                                         base_scene=base_scene, obj_src=obj_src)
            cue_controlers.append(cue_ctrl)

        for pstrength in args.cue_strength:
            for cue_controler in cue_controlers:
                lgr.info("Adding cue: %s (%sX)", cue_controler, pstrength)
                renderable = cue_controler.add_cue(renderable, strength=pstrength)

            render_result = renderable.render()
            if CUE_SATURATION in args.cue:
                render_result = enhaze(render_result, strength=pstrength)
            fname_stem = format_filename_stem(args, pobj)
            dscale = write_results(render_result, fname_stem, run_name)
            if DEBUG and args.save_state:
                renderable.renderer.core.save_state(f"output/debug_{fname_stem}.blend")
            records.append(_record(args, penv, pobj, fname_stem, dscale, cue_strength=pstrength))

            for cue_controler in cue_controlers[::-1]:
                lgr.info("Removing cue: %s (%sX)", cue_controler, pstrength)
                renderable = cue_controler.remove_cue(renderable, strength=pstrength)

            if len(args.cue) == 1 and args.cue[0] == CUE_FAMILIAR_SIZE:
                break  # FS does not support cue strength

    save_records(records, run_name)


def _base_scene_params(args):
    if CUE_FAMILIAR_SIZE in args.cue:
        return itertools.product(
            SELECTED_ENVIRONMENTS[:1] if DEBUG else SELECTED_ENVIRONMENTS,
            FAMILIAR_SIZE_OBJECT_PAIRS[:1] if DEBUG else FAMILIAR_SIZE_OBJECT_PAIRS,
        )

    return itertools.product(
        SELECTED_ENVIRONMENTS[:1] if DEBUG else SELECTED_ENVIRONMENTS,
        SELECTED_OBJECTS[:1] if DEBUG else SELECTED_OBJECTS,
    )


def setup_base_scene(args, _scene, obj_src, target_name, small_targ_name=None):
    if CUE_FAMILIAR_SIZE in args.cue:
        if args.position == ODD_DEPTH_FAR:
            # odd at far, so target being the default larger one
            target_name, distractor_name = target_name, small_targ_name
        else:
            target_name, distractor_name = small_targ_name, target_name
        base_scene_builder = FamiliarSizeKbBaseSceneBuilder(
            target_name, distractor_name=distractor_name,
            n_distractors=args.n_distractors, odd_scaler=args.odd_scaler
        )

    else:
        base_scene_builder = KbBaseSceneBuilder(
            target_name, n_distractors=args.n_distractors, odd_scaler=args.odd_scaler)

    base_scene = base_scene_builder.build(obj_src=obj_src, kb_scene=_scene)

    base_scene = base_scene_builder.add_oddity(base_scene, args.position, obj_src=obj_src)
    return base_scene


def setup_renderable(args, penv, base_scene, renderer, kubasic):
    env_type, env_id = penv

    env_builder = get_env_builder(
        env_type, base_scene.kb_scene,
        kb_renderer=renderer.core,
        kubasic=kubasic,  # HDRI needs kubasic dome
    )
    renderable = KbRenderable(base_scene, renderer, env_builder=env_builder)

    if args.elim_lp:  # only supported by KbMinEnvBuilder
        renderable.setup_environment(env_id=env_id, textureless_ground=True)
    else:
        renderable.setup_environment(env_id=env_id)
    return renderable


def get_env_builder(
    builder_name,
    kb_scene,
    kb_renderer=None,
    kubasic=None,
):
    if builder_name == BASIC_ENV_TYPE:
        return KbMinEnvBuilder(kb_scene)
    if builder_name == HDRI_ENV_TYPE:
        hdri_source = kb.AssetSource.from_manifest(args.hdri_assets)
        dome = kubasic.create(
            asset_id="dome", name="dome", static=True, background=True,
            position=(0, 0, -0.001), # avoid bottom cutoff
            segmentation_id=OTHER_SEG_ID)
        return KbHdriEnvBuilder(kb_scene, kb_renderer, hdri_source, dome=dome)

    raise ValueError(f"Unsupported builder name: {builder_name}")


def get_cue_controler(pcue, odd_scaler, base_scene=None, obj_src=None):
    if base_scene.target.position[1] > base_scene.distractors[0].position[1]:
        odd_position = ODD_DEPTH_FAR
    else:
        odd_position = ODD_DEPTH_NEAR

    if pcue == CUE_TEXTURE_GRADIENT:
        freq = 5 / _obj_dims(base_scene.target)[0]
        targ_obj_tex_op = TargetObjTextureOperator(freq, odd_position=odd_position)
        dist_obj_tex_op = DistractorObjTextureOperator(freq, odd_position=odd_position)
        return KbCueControler(scene_operators=[targ_obj_tex_op, dist_obj_tex_op])

    if pcue == CUE_OCCLUSION:
        cam_d = base_scene.camera.position[1]
        targ_d, dist_d = base_scene.target.position[1], base_scene.distractors[0].position[1]
        if odd_position == ODD_DEPTH_FAR:
            obj_width = _obj_dims(base_scene.distractors[0])[0]
            far_depth = cam_d - targ_d
            near_depth = cam_d - dist_d
        else:
            obj_width = _obj_dims(base_scene.target)[0]
            far_depth = cam_d - dist_d
            near_depth = cam_d - targ_d

        odd_scaler_ = far_depth / near_depth
        lgr.info("Scaler diff for OC: %s", odd_scaler - odd_scaler_)

        if not math.isclose(odd_scaler, odd_scaler_, rel_tol=1e-3): # for FS cue strength=1
            x_gap = abs(base_scene.distractors[0].position[0] - base_scene.target.position[0])
            if odd_position == ODD_DEPTH_FAR:
                base_dist = x_gap * odd_scaler_/(odd_scaler_-1)
                unit_dist = -(odd_scaler_-1)*0.8*obj_width
            else:
                base_dist = x_gap / (odd_scaler_-1)
                unit_dist = -(odd_scaler_-1)*0.8*obj_width
        else:
            base_dist = odd_scaler_ * obj_width  # tangent
            unit_dist = odd_scaler_/(odd_scaler_-1) * obj_width / MAX_CUE_STRENGTH/2  # half OC @ max
        cam_pos_op = CameraPositionOperator(
            axis=0, unit_dist=unit_dist, base_dist=base_dist, relative=True)
        return KbCueControler(scene_operators=[cam_pos_op])

    if pcue == CUE_HEIGHT_IN_PLANE:
        unit_dist = _obj_dims(base_scene.target)[2] / odd_scaler
        cam_pos_op = CameraPositionOperator(axis=2, unit_dist=unit_dist, relative=True)
        return KbCueControler(scene_operators=[cam_pos_op])

    if pcue == CUE_SHADOW:
        horizonal_offset = abs(base_scene.target.position[0] - base_scene.distractors[0].position[0])
        if odd_position == ODD_DEPTH_NEAR:
            offset_multiplier = 1 / (odd_scaler-1)
        else:
            offset_multiplier = odd_scaler / (odd_scaler-1)
        light_pos_x = -horizonal_offset * offset_multiplier

        cam_d = base_scene.camera.position[1]
        obj_d = base_scene.target.position[1]
        obj_h = _obj_dims(base_scene.target)[2]
        light_op = LightOperator(position=(light_pos_x, cam_d, obj_h),
                                 look_at=(0, obj_d, 0.75*obj_h))
        return KbCueControler(scene_operators=[light_op])

    if pcue == CUE_FOCUS:
        cam_dof_op = CameraDOFOperator()
        return KbCueControler(scene_operators=[cam_dof_op])

    if pcue == CUE_RELATIVE_SIZE:
        scene_ops = [
            ScaleOperator("target"),
            ObjectPlacementOperator("target", obj_src),
        ]
        return KbRelativeSizeCueControler(scene_operators=scene_ops,
                                          odd_scaler=odd_scaler,
                                          odd_position=odd_position, strength_factor=1)

    if pcue == CUE_SATURATION:  # will be post-processed
        return KbCueControler(scene_operators=[rawdata.SceneNoOp()])

    if pcue == CUE_FAMILIAR_SIZE:  # done while building base scene
        return KbCueControler(scene_operators=[rawdata.SceneNoOp()])

    raise ValueError(f"Unsupported cue param: {pcue}")


def enhaze(render_result: rawdata.RenderResult, strength=1.0) -> rawdata.RenderResult:
    rgb_img = render_result.image[:, :, :3]
    depth_img = render_result.depth[:, :, 0]  # shape was (n, n, 1)
    scatter_coef = strength / 100
    enhazed = haze.apply_haze_eq(rgb_img, depth_img, scatter_coef=scatter_coef)
    render_result.image = enhazed
    return render_result


def format_run_name() -> str:
    return f"kb{timestamp_str(compact=True)}"


def format_out_dirpath(run_name, subdir) -> pathlib.Path:
    if DEBUG:
        out_dirpath = pathlib.Path(__file__).parent/"output"/subdir
    else:
        out_dirpath = DATA_DIRPATH/"kubric_scenes"/run_name/subdir/"orig"
    return out_dirpath


def format_filename_stem(args, pobj) -> str:
    if DEBUG:
        return "_".join(pobj) if isinstance(pobj, tuple) else "_".join(pobj[0])

    return "_".join((timestamp_str(compact=True),
                    args.position,
                    "".join(args.cue),
                    ))


def format_out_csv_path(run_name):
    dirpath = prep_out_dirpath(run_name).parent.parent
    csv_out_path = dirpath/f"{run_name}.csv"
    return csv_out_path


def prep_out_dirpath(run_name, subdir="images") -> pathlib.Path:
    out_dirpath = format_out_dirpath(run_name, subdir)

    out_dirpath.mkdir(exist_ok=True, parents=True)
    return out_dirpath


def write_results(render_result: rawdata.RenderResult, fname_stem, run_name):
    images_dir = prep_out_dirpath(run_name)
    dpmaps_dir = prep_out_dirpath(run_name, subdir="dpmaps")
    tlabel_dir = prep_out_dirpath(run_name, subdir="targ_labels")
    dlabel_dir = prep_out_dirpath(run_name, subdir="dist_labels")
    tsegmt_dir = prep_out_dirpath(run_name, subdir="targ_segmts")
    dsegmt_dir = prep_out_dirpath(run_name, subdir="dist_segmts")

    kb.write_png(render_result.image, images_dir/f"{fname_stem}.png")
    kb.write_palette_png(render_result.targ_labels, tlabel_dir/f"{fname_stem}.png")
    kb.write_palette_png(render_result.dist_labels, dlabel_dir/f"{fname_stem}.png")
    dscale = kb.write_scaled_png(render_result.depth, dpmaps_dir/f"{fname_stem}.png")
    _save_array_as_png(render_result.tsegmt, tsegmt_dir/f"{fname_stem}.png")
    _save_array_as_png(render_result.dsegmt, dsegmt_dir/f"{fname_stem}.png")
    return dscale


def _record(args, penv, pobj, fname_stem, dscale, cue_strength=None):
    return dict(image_name=f"{fname_stem}.png",
                env_cat=penv[0], env_id=penv[1],
                obj_cat=pobj[0], obj_id=pobj[1],
                odd_position=args.position,
                cues=args.cue,
                cue_strength=cue_strength,
                depth_scale=dscale,
                )


def timestamp_str(compact=False):
    if compact:
        return dt.datetime.now().strftime("%y%m%d%H%M%S")
    return dt.datetime.now().replace(microsecond=0).isoformat()


def _save_array_as_png(array, path):
    array_rescaled = array * 255 // array.max()  # seg_id -> grey scale, for easier visualization
    array_uint8 = array_rescaled.astype(np.uint8)  # max = 255
    lgr.debug("array_uint8 min/max: %s/%s", array_uint8.min(), array_uint8.max())
    array_2d = array_uint8.squeeze()
    PIL.Image.fromarray(array_2d).save(path)


def save_records(rec_df, run_name):
    rec_df = pd.DataFrame.from_records(rec_df)
    csv_out_path = format_out_csv_path(run_name)
    rec_df.to_csv(csv_out_path, index=False)
    lgr.info("Saved records to %s", csv_out_path)
    return


def validate_args(args):
    invalid_args = (
        CUE_HEIGHT_IN_PLANE in args.cue and CUE_HEIGHT_IN_PLANE_CTRL in args.cue,
    )
    assert not any(invalid_args)

    if CUE_TEXTURE_GRADIENT in args.cue:  # any mutation to obj will reset TG; moving to the last
        args.cue.remove(CUE_TEXTURE_GRADIENT)
        args.cue.append(CUE_TEXTURE_GRADIENT)


## KB util functions
def _obj_dims(obj):
    """Assuming no rotation"""
    xs = obj.bbox_3d[:, 0]
    ys = obj.bbox_3d[:, 1]
    zx = obj.bbox_3d[:, 2]
    return xs.max()-xs.min(), ys.max()-ys.min(), zx.max()-zx.min()


def _name_to_id(name):
    return name.replace("_copy", "")


if __name__ == "__main__":
    parser = kb.ArgumentParser()

    # Kubric Configuration
    parser.add_argument("--kubasic_assets", type=str,
                        default="resources/KuBasic.json")
    parser.add_argument("--hdri_assets", type=str,
                        default="resources/HDRI_haven.json")
    parser.add_argument("--gso_assets", type=str,
                        default="resources/GSO.json")
    parser.add_argument("--save_state", dest="save_state", action="store_true")

    # odd-one-out config
    parser.add_argument("-n", "--n_distractors", help="number of distractor objects",
                        type=int)
    parser.add_argument("-S", "--odd_scaler", help="scaler for the odd/target object",
                        type=float, default=ODD_OBJ_SCALER)  # overwritten by randomization in main()

    parser.add_argument("-p", "--position", help="odd object position",
                        choices=[ODD_NONE, ODD_DEPTH_FAR, ODD_DEPTH_NEAR],
                        default=ODD_DEPTH_FAR)

    # cue args
    parser.add_argument("-c", "--cue", help="depth cue to test",
                        nargs="*", choices=CUE_CHOICES)
    parser.add_argument("-C", "--secondary_cue", help="list of cues to test 2nd-deg interaction",
                        nargs="*", choices=CUE_CHOICES)
    parser.add_argument("-L", "--elim_lp", help="Eliminate LP cue",
                        action="store_true")

    parser.add_argument("-s", "--cue_strength", help="strength of each depth cue",
                        type=float, nargs="+")
    parser.add_argument("-i", "--incremental_cue", help="add cue in args.cue one by one",
                        action="store_true")
    parser.add_argument("-I", "--interaction_degree", help="number of cues for interaction",
                        type=int, default=0)  # by default add all args.cue

    parser.set_defaults(save_state=False, resolution=256 if DEBUG else 1024,
                        n_distractors=4,
                        cue=[],
                        cue_strength=DEFAULT_CUE_STRENGTH_VALUES,
                        )

    args = parser.parse_args()

    try:
        if args.incremental_cue:
            lgr.info("Gradually adding cues: %s", args.cue)
            all_cues = args.cue
            curr_cues = []
            args.cue = curr_cues
            main(args)
            for cue in all_cues:
                curr_cues.append(cue)
                args.cue = curr_cues
                main(args)

        elif args.interaction_degree:
            lgr.info("%s-degree interaction among cues: %s", args.interaction_degree, args.cue)
            all_cues = args.cue[:]
            for curr_cues in sorted(itertools.combinations(all_cues, args.interaction_degree)):
                args.cue = list(curr_cues)
                main(args)

        elif args.secondary_cue:
            lgr.info("Creating images for 2nd-deg interaction: %s", args.secondary_cue)
            base_cue = args.cue[:]
            for cue in args.secondary_cue:
                args.cue = base_cue + [cue]
                main(args)

        else:
            main(args)

    except Exception:
        lgr.exception("Unexpected error!")

"""
sudo docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$PWD:/kubric" \
    --mount type=bind,source=$PWD/..,target=/o3d_src \
    --env PYTHONPATH="${PYTHONPATH}:/o3d_src" \
    --mount type=bind,source=$PWD/../../data,target=$PWD/../../data \
    --env O3D_DATA_DIRPATH="$PWD/../../data" \
    --env DEBUG=1 \
    kubricdockerhub/kubruntu \
    python3 main.py \
    -p far -c HP
"""
