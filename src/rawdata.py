"""O3-D dataset generation"""

import abc
import dataclasses
import typing as T

import numpy as np

CameraT = T.Any
TargetT = T.Any
DistractorT = TargetT
LightT = T.Any
GroundT = T.Any
BackgroundT = T.Any


@dataclasses.dataclass
class O3dSceneAbc:
    """Info of scene states."""
    target_name: str
    n_distractors: int
    # below for quick access
    camera: CameraT=None
    target: TargetT=None
    distractors: T.List[DistractorT]=None
    light: LightT=None
    ground: GroundT=None
    background: BackgroundT=None

    @classmethod
    @abc.abstractmethod
    def from_blend(cls, path) -> "O3dSceneAbc":
        pass

    @abc.abstractmethod
    def to_blend(self, path):
        pass


@dataclasses.dataclass
class RenderResult:
    image: np.ndarray
    segmt: np.ndarray=None
    depth: np.ndarray=None
    targ_labels: np.ndarray=None
    dist_labels: np.ndarray=None


class BaseSceneBuilderAbc(abc.ABC):
    target_name: str
    distractor_name: str=None
    n_distractors: int=4
    odd_scaler: float=1.2

    def __init__(self, target_name, distractor_name=None,
                 odd_scaler=1.2, n_distractors=4):
        self.target_name = target_name
        self.distractor_name = distractor_name or target_name
        self.n_distractors = n_distractors
        self.odd_scaler = odd_scaler

    @abc.abstractmethod
    def build(self, **kwargs) -> O3dSceneAbc:
        pass

    @abc.abstractmethod
    def add_oddity(self, scene: O3dSceneAbc, position: str, **kwargs) -> O3dSceneAbc:
        pass


class RendererAbc(abc.ABC):
    @abc.abstractmethod
    def render(self, scene: O3dSceneAbc) -> RenderResult:
        pass


class Renderable(abc.ABC):
    """A generic class to handle inter-dependent scene and renderer."""
    scene: O3dSceneAbc
    renderer: RendererAbc

    def __init__(self, scene: O3dSceneAbc, renderer: RendererAbc):
        self.scene = scene
        self.renderer = renderer

    def render(self) -> RenderResult:
        return self.renderer.render(self.scene)


class ISceneOperator(abc.ABC):
    @abc.abstractmethod
    def change(self, scene: O3dSceneAbc, scale=1.0) -> O3dSceneAbc:
        pass

    @abc.abstractmethod
    def unchange(self, scene: O3dSceneAbc, scale=1.0) -> O3dSceneAbc:
        pass


class IRendererOperator(abc.ABC):
    @abc.abstractmethod
    def operate(self, renderer: RendererAbc):
        pass


class SceneNoOp(ISceneOperator):
    def change(self, scene, scale=1.0):
        return scene

    def unchange(self, scene, scale=1.0):
        return scene


class RendererNoOp(IRendererOperator):
    def operate(self, renderer):
        return renderer


class CueControlerAbc(abc.ABC):
    scene_operators: T.List[ISceneOperator]=None
    renderer_operator: IRendererOperator=RendererNoOp()

    def __init__(self, scene_operators=None,
                 renderer_operator=RendererNoOp(),
                 **kwargs):
        self.scene_operators = scene_operators or [SceneNoOp()]
        self.renderer_operator = renderer_operator

    @abc.abstractmethod
    def add_cue(self, renderable: Renderable, strength=1.0) -> Renderable:
        pass

    @abc.abstractmethod
    def remove_cue(self, renderable: Renderable, strength=1.0) -> RendererAbc:
        pass


## Level 2 classes, Kubric
# see imggen/main.py for implementation