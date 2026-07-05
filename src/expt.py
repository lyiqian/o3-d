import abc
import logging
import pathlib
import random
import re
import typing

import pandas as pd
import datasets

import core
from core import IclDemo
import util


Q_ = core.Question

DESCRIPTION_QUESTIONS = [
    Q_("What object is unique in this image?", kind="description",
                  clarity=Q_.CLARITY_LOW),
    Q_("Describe the most interesting object in this image.", kind="description",
                  clarity=Q_.CLARITY_LOW),
    Q_("Describe the most eye-catching object in this image.", kind="description",
                  clarity=Q_.CLARITY_LOW),
    Q_("Describe the salient object in this image.", kind="description",
                  clarity=Q_.CLARITY_LOW),
]

HIGH_CLARITY_TARGET_REFERENCES = [
    "In this image, one object is in the middle of other objects. ",
    # "Answer a question about the middle object. ",
    "Answer a question about the left object. ",
    "Answer a question about the right object. ",
]
DEPTH_ORDERING_QUESTIONS = [
    "Is the object closer to the viewer or farther away than the other similar objects? ",
    "Is the object behind the other similar objects? ",
]

METADATA_FILENAME = 'metadata.jsonl'

lgr = logging


class BaseImageSet(abc.ABC):
    NAME: str = None
    SUBDIR_NAME: str = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abc.abstractmethod
    def list_image_names(self) -> typing.List[str]:
        pass

    @abc.abstractmethod
    def get_image_path(self, image_name: str, **kwargs) -> str:
        pass

    @abc.abstractmethod
    def get_image(self, image_name: str) -> core._RegularImage:
        pass


class BaseQuestionSet(abc.ABC):
    NAME: str = None

    @abc.abstractmethod
    def list_questions(self, **kwargs) -> typing.List[Q_]:
        pass


## Abs Level 2 classes
class _DirectoryImageSet(BaseImageSet):
    NAME = None  # to be implemented
    SUBDIR_NAME = None  # to be implemented
    IMG_EXT = "jpg"

    DATA_PATH = util.DEFAULT_DATA_PATH

    def list_image_names(self):
        img_dirpath = self.DATA_PATH / self.SUBDIR_NAME
        img_names = [img_path.name for img_path in img_dirpath.glob(f"*.{self.IMG_EXT}")]

        if self.kwargs.get("name_contains"):
            img_names = [n for n in img_names if self.kwargs.get("name_contains") in n]

        return sorted(img_names)

    @classmethod
    def get_image_path(cls, image_name: str, ignore_nonexist=False, **kwargs) -> str:
        """Supporting aug img w/o aug suffix."""
        image_path = cls.DATA_PATH/cls.SUBDIR_NAME/image_name
        if ignore_nonexist:
            return str(image_path)

        if not image_path.exists():
            image_dirpath = image_path.parent
            prefix = image_path.stem
            matched_paths = list(image_dirpath.glob(f"{prefix}*.{cls.IMG_EXT}"))

            assert len(matched_paths) == 1, f"Cannot uniquely locate path for {image_name}."
            image_path = matched_paths[0]
            lgr.debug("Got image path by prefix matching for %s", image_name)

        return str(image_path)

    def get_image(self, image_name: str):
        image_path = self.get_image_path(image_name)
        return core.FileSysImage(image_path, self.NAME)

    def __len__(self):
        return len(self.list_image_names())


class _KubricDirectoryImageSet(_DirectoryImageSet):
    IMG_EXT = "png"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # init self.info df
        info_dirpath = (util.DEFAULT_DATA_PATH/self.SUBDIR_NAME).parent.parent
        info_filepath = info_dirpath / METADATA_FILENAME
        self.info = pd.read_json(info_filepath, lines=True)

    def list_image_names(self):
        image_names = super().list_image_names()
        if util.DEBUG:
            image_names = image_names[:3]
        return image_names


### For (pre-huggingface) File System based Kubric image sets ###
class KubricOneCueAugmentedImageSet(_KubricDirectoryImageSet):
    """HF subset: kb-1cue"""
    NAME = "kb-1cue-aug"
    SUBDIR_NAME = "kubric_scenes/one_cue/images/augmented"

class KubricOneCueMarkedImageSet(_KubricDirectoryImageSet):
    """Generated via dataaug.py"""
    NAME = "kb-1cue-mark"
    SUBDIR_NAME = "kubric_scenes/one_cue/images/marked"


class KubricTwoCueAugmentedImageSet(_KubricDirectoryImageSet):
    """HF subset: kb-2cue"""
    NAME = "kb-2cue-aug"
    SUBDIR_NAME = "kubric_scenes/two_cue/images/augmented"

class KubricTwoCueMarkedImageSet(_KubricDirectoryImageSet):
    """Generated via dataaug.py"""
    NAME = "kb-2cue-mark"
    SUBDIR_NAME = "kubric_scenes/two_cue/images/marked"


class KubricNoCueAugmentedImageSet(_KubricDirectoryImageSet):
    """HF subset: kb-0cue"""
    NAME = "kb-0cue-aug"
    SUBDIR_NAME = "kubric_scenes/no_cue/images/augmented"

class KubricNoCueMarkedImageSet(_KubricDirectoryImageSet):
    """Generated via dataaug.py"""
    NAME = "kb-0cue-mark"
    SUBDIR_NAME = "kubric_scenes/no_cue/images/marked"


class KubricNoLpAugmentedImageSet(_KubricDirectoryImageSet):
    """HF subset: kb-no-lp"""
    NAME = "kb-no-lp-aug"
    SUBDIR_NAME = "kubric_scenes/no_lp/images/augmented"

class KubricNoLpMarkedImageSet(_KubricDirectoryImageSet):
    """Generated via dataaug.py"""
    NAME = "kb-no-lp-mark"
    SUBDIR_NAME = "kubric_scenes/no_lp/images/marked"


#### For HuggingFace datasets
class _HuggingFaceImageSet(BaseImageSet):
    NAME = None  # to be implemented
    SUBDIR_NAME = None  # for formatting image path only
    IMG_EXT = "png"

    HF_DATASET_NAME = "liuyiqian/O3-D"
    HF_SUBSET_NAME = None  # to be implemented
    HF_SPLIT = "train"  # default; did not create other splits

    IMAGE_LIKE_COLUMNS = ['image']
    MAIN_IMAGE_COLUMN = 'image'

    def __init__(self, **kwargs):
        self.ds = datasets.load_dataset(
            self.HF_DATASET_NAME, self.HF_SUBSET_NAME,
            split=self.HF_SPLIT,
            cache_dir=util.HF_CACHE,
            # todo consider streaming=True; won't work for ICL demo sampling
        )
        self.info = self.ds.remove_columns(self.IMAGE_LIKE_COLUMNS).to_pandas()

        self.image_format = kwargs.get('image_format')
        if self.image_format == 'bytes':
            for col in self.IMAGE_LIKE_COLUMNS:
                self.ds = self.ds.cast_column(col, datasets.Image(decode=False))
        else:
            self.ds.set_format(type=self.image_format, columns=self.IMAGE_LIKE_COLUMNS)

    def list_image_names(self):
        return self.ds["image_name"]

    def get_image_path(self, image_name: str, **kwargs):
        """format a image path consistent with pre-HF ImageSets"""
        if self.SUBDIR_NAME is None:
            raise NotImplementedError
        return str(util.DEFAULT_DATA_PATH/self.SUBDIR_NAME/image_name)

    def get_image(self, image_name: str) -> core.HuggingFaceImage:
        if self.image_format == 'bytes':
            return core.HuggingFaceBytesImage(
                image_name, self.NAME,
                hf_dataset=self.ds, img_col=self.MAIN_IMAGE_COLUMN)
        else:
            return core.HuggingFaceImage(
                image_name, self.NAME,
                hf_dataset=self.ds, img_col=self.MAIN_IMAGE_COLUMN)

    def is_marked(self):
        return self.NAME.endswith("-mark")

    def __len__(self):
        return len(self.ds)


class HfKbNoCueImageSet(_HuggingFaceImageSet):
    NAME = "hf-kb-0cue"
    HF_SUBSET_NAME = "kb-0cue"
    SUBDIR_NAME = KubricNoCueAugmentedImageSet.SUBDIR_NAME

    IMAGE_LIKE_COLUMNS = ['image', 'depth_map', 'targ_seg', 'dist_seg']

class HfKbOneCueImageSet(_HuggingFaceImageSet):
    NAME = "hf-kb-1cue"
    HF_SUBSET_NAME = "kb-1cue"
    SUBDIR_NAME = KubricOneCueAugmentedImageSet.SUBDIR_NAME

    IMAGE_LIKE_COLUMNS = ['image', 'depth_map', 'targ_seg', 'dist_seg']

class HfKbTwoCueImageSet(_HuggingFaceImageSet):
    NAME = "hf-kb-2cue"
    HF_SUBSET_NAME = "kb-2cue"
    SUBDIR_NAME = KubricTwoCueAugmentedImageSet.SUBDIR_NAME

    IMAGE_LIKE_COLUMNS = ['image', 'depth_map', 'targ_seg', 'dist_seg']

class HfKbNoLpImageSet(_HuggingFaceImageSet):
    NAME = "hf-kb-no-lp"
    HF_SUBSET_NAME = "kb-no-lp"
    SUBDIR_NAME = KubricNoLpAugmentedImageSet.SUBDIR_NAME

    IMAGE_LIKE_COLUMNS = ['image', 'depth_map', 'targ_seg', 'dist_seg']


class HfReal012CueImageSet(_HuggingFaceImageSet):
    NAME = "hf-real-012cue"
    HF_SUBSET_NAME = "real-012cue"

    IMAGE_LIKE_COLUMNS = ['image', 'augmented_image', 'marked_image', 'targ_seg', 'dist_seg']
    MAIN_IMAGE_COLUMN = 'image'

class HfReal012CueAugmentedImageSet(HfReal012CueImageSet):
    NAME = "hf-real-012cue-aug"
    MAIN_IMAGE_COLUMN = 'augmented_image'

class HfReal012CueMarkedImageSet(HfReal012CueImageSet):
    NAME = "hf-real-012cue-mark"
    MAIN_IMAGE_COLUMN = 'marked_image'


class HfReal012CueCroppedAugmentedImageSet(_HuggingFaceImageSet):
    NAME = "hf-real-012cue-cropped-aug"
    HF_SUBSET_NAME = "real-012cue-cropped"

    IMAGE_LIKE_COLUMNS = ['augmented_image', 'marked_image', 'targ_seg', 'dist_seg']
    MAIN_IMAGE_COLUMN = 'augmented_image'

class HfReal012CueCroppedMarkedImageSet(_HuggingFaceImageSet):
    NAME = "hf-real-012cue-cropped-mark"
    HF_SUBSET_NAME = "real-012cue-cropped"

    IMAGE_LIKE_COLUMNS = ['augmented_image', 'marked_image', 'targ_seg', 'dist_seg']
    MAIN_IMAGE_COLUMN = 'marked_image'


class HfRealMixedCueImageSet(_HuggingFaceImageSet):
    NAME = "hf-real-mcue"
    HF_SUBSET_NAME = "real-mcue"

    IMAGE_LIKE_COLUMNS = ['image', 'marked_image', 'targ_seg', 'dist_seg']
    MAIN_IMAGE_COLUMN = 'image'

class HfRealMixedCueMarkedImageSet(HfRealMixedCueImageSet):
    NAME = "hf-real-mcue-mark"
    MAIN_IMAGE_COLUMN = 'marked_image'


#### Question Set classes
class HfVisualQuestionSet(BaseQuestionSet):
    """A special HF dataset subset called visual_questions, having cols described below.

    - subset_name == HF_SUBSET_NAME, in the above _HuggingFaceImageSet classes
    - image_name == ds['image_name']
    - is_marked -> XXXMarkedImageSet, otherwise the augmented image is used
    - question: the sampled question; see _RandTemplateQuestionSet for sampling
    - ques_clarity == core.QUESTION.clarity
    """

    NAME = "hf-visual-questions"
    KIND = "closer_farther"
    CLARITY = None  # mixed, determined by the ques_clarity column

    HF_DATASET_NAME = "liuyiqian/O3-D"
    HF_SUBSET_NAME = "visual_questions"
    HF_SPLIT = "train"  # default; did not create other splits

    def __init__(self):
        vq_ds = datasets.load_dataset(
            self.HF_DATASET_NAME, self.HF_SUBSET_NAME,
            split=self.HF_SPLIT,
            cache_dir=util.HF_CACHE,
        )
        self.vq_df = (
            vq_ds.to_pandas()
                .set_index(["subset_name", "image_name", "is_marked"])
                .sort_index()
        )

    def list_questions(self, **kwargs):
        """List questions given image information"""
        subset_name = kwargs["subset_name"]
        image_name = kwargs["image_name"]
        is_marked = kwargs.get("is_marked", True)

        filtered_df = self.vq_df.loc[(subset_name, image_name, is_marked)]
        questions = [
            Q_(text=row.question, kind=self.KIND, clarity=row.ques_clarity)
            for row in filtered_df.itertuples(index=False)
        ]
        return questions

    @classmethod
    def prep_question_kwargs(cls, image_set: _HuggingFaceImageSet, image_name: str):
        return {"subset_name": image_set.HF_SUBSET_NAME,
                "image_name": image_name,
                "is_marked": image_set.is_marked()}


class _TemplateQuestionSet(BaseQuestionSet):
    NAME = None  # to be implemented
    KIND = None  # to be implemented
    CLARITY = None  # to be implemented

    def list_questions(self, **kwargs):
        context = self._get_context()
        targ_refs = self._list_targ_refs()
        query = self.get_query()
        questions = [
            Q_(text=f"{context}{targ_ref}{query}", kind=self.KIND, clarity=self.CLARITY)
            for targ_ref in targ_refs
        ]
        return questions

    def _get_context(self) -> str:
        return ""

    @abc.abstractmethod
    def _list_targ_refs(self) -> typing.List[str]:
        pass

    @abc.abstractmethod
    def get_query(self, response_format=True) -> str:
        pass

    @abc.abstractmethod
    def _get_response_format(self) -> str:
        pass


class LegacyDepthOrderQuestionSet(_TemplateQuestionSet):
    NAME = None
    KIND = "closer_farther"
    CLARITY = None  # to be implemented

    def _list_targ_refs(self):
        raise NotImplementedError

    def get_query(self, response_format=True):
        ques = random.choice(DEPTH_ORDERING_QUESTIONS)

        if response_format:
            return ques + self._get_response_format(ques)
        else:
            return ques + "Answer with one sentence."

    def _get_response_format(self, ques):
        if "closer" in ques:
            return f"{_rand_mcq(['closer', 'farther'])}Answer A or B."
        elif "behind" in ques:
           return f"{_rand_mcq(['Yes', 'No'])}Answer A or B."
        else:
            raise ValueError(f"Unsupported ques for resp fmt: {ques}")


class DepthOrderHigh35ClarityQuestionSet(LegacyDepthOrderQuestionSet):
    NAME = "depth-order-highc"
    CLARITY = Q_.CLARITY_HIGH

    def _list_targ_refs(self):
        return [HIGH_CLARITY_TARGET_REFERENCES[0]]  # middle object

class DepthOrderHigh35ClarityLeftQuestionSet(LegacyDepthOrderQuestionSet):
    NAME = "depth-order-highc-l"
    CLARITY = Q_.CLARITY_HIGH

    def _list_targ_refs(self):
        return [HIGH_CLARITY_TARGET_REFERENCES[1]]  # left object

class DepthOrderHigh35ClarityRightQuestionSet(LegacyDepthOrderQuestionSet):
    NAME = "depth-order-highc-r"
    KIND = "closer_farther"
    CLARITY = Q_.CLARITY_HIGH

    def _list_targ_refs(self):
        return [HIGH_CLARITY_TARGET_REFERENCES[2]]  # right object

class DepthOrderHigh35ClarityLRQuestionSet(BaseQuestionSet):
    NAME = "depth-order-highc-lr"

    def list_questions(self, **kwargs):
        return (
            DepthOrderHigh35ClarityLeftQuestionSet().list_questions() +
            DepthOrderHigh35ClarityRightQuestionSet().list_questions()
        )


class MinQuestionSet(BaseQuestionSet):
    NAME = "min"
    def list_questions(self, **kwargs):
        return DESCRIPTION_QUESTIONS[:1]


### FOR ECCV PAPER ###
class _RandTemplateQuestionSet(BaseQuestionSet):
    """Sample a (string) template to format the question.

    Any template has two kw: {target} & {distractors}.
    """
    NAME = None  # to be implemented
    KIND = None  # to be implemented
    CLARITY = None # to be implemented

    TEMPLATES = None  # to be implemented

    def list_questions(self, **kwargs):
        """As this is for randomized questions, each func call only return one Q."""
        template = self._sample_template()

        target_ref = self._sample_target_ref()
        distractors_ref = self._sample_distractors_ref()
        question_text = self._format_question_text(template, target_ref, distractors_ref)
        question = Q_(text=question_text, kind=self.KIND, clarity=self.CLARITY)
        return [question]


    def _sample_template(self) -> tuple:
        """Return 2-tuple of question template and MCQ option list."""
        template = random.choice(self.TEMPLATES)
        return template

    @abc.abstractmethod
    def _sample_target_ref(self) -> str:
        pass

    @abc.abstractmethod
    def _sample_distractors_ref(self) -> str:
        pass

    @abc.abstractmethod
    def _format_question_text(self, template, target_ref, distractors_ref) -> str:
        pass


class _DepthOrderRandQSet(_RandTemplateQuestionSet):
    NAME = None  # to be implemented
    KIND = "closer_farther"
    CLARITY = None # to be implemented

    TEMPLATES = [
        # A or B questions
        ("Is {target} closer to or farther away from the viewer than {distractors}?", ["Closer", "Farther"]),
        ("Is {target} positioned farther from or closer to the observer than {distractors}?", ["Closer", "Farther"]),
        ("Is {target} behind or in front of {distractors}?", ["Behind", "In front of"]),
        ("Relative to the camera, is {target} closer or farther away than {distractors}?", ["Closer", "Farther"]),
        ("Does {target} come before or after {distractors}?", ["Before", "After"]),
        ("Does {target} feel farther or nearer than {distractors}?", ["Nearer", "Farther"]),
        ("Along the line of sight, is {target} positioned before or after {distractors}?", ["Before", "After"]),
        # Yes or No questions
        ("Is {target} at the rear of {distractors}?", ["Yes", "No"]),
        ("True or False: {target} is located in front of {distractors}.", ["True", "False"]),
    ]

    TARGET_REFS = None  # to be implemented
    DISTRACTORS_REFS = [
        "the other objects",
        "the other similar objects",
        "the rest of the objects",
        "the remaining objects",
    ]

    def _sample_target_ref(self) -> str:
        return random.choice(self.TARGET_REFS)

    def _sample_distractors_ref(self) -> str:
        return random.choice(self.DISTRACTORS_REFS)


    def _format_question_text(self, template, target_ref, distractors_ref):
        body_template, options = template

        body = body_template.format(target=target_ref, distractors=distractors_ref)

        resp_format = f"{_rand_mcq(options)}Answer A or B."

        return f"{body} {resp_format}"

class DepthOrderLowClarityRandQSet(_DepthOrderRandQSet):
    NAME = "depth-order-lowc-rand"
    CLARITY = Q_.CLARITY_LOW

    TARGET_REFS = [
        "the salient object",
        "the object that stands out",
        "the uniquely positioned object",
        "the special object",
        "the most interesting object",
        "the object that is different",
        "the visually distinct object",
    ]


class DepthOrderMedClarityRandQSet(_DepthOrderRandQSet):
    NAME = "depth-order-medc-rand"
    CLARITY = Q_.CLARITY_MEDIUM

    TARGET_REFS = [
        "the salient object due to its unique distance in the image",
        "the object that stands out because it is at a different depth",
        "the object positioned at a different distance",
        "the special object breaking the depth pattern",
        "the object that is different from the rest in terms of its depth",
    ]


class DepthOrderHighClarityRandQSet(_DepthOrderRandQSet):
    NAME = "depth-order-highc-rand"
    CLARITY = Q_.CLARITY_HIGH

    TARGET_REFS = [
        "the object in the middle",
        "the center object",
    ]

class DepthOrderHighestClarityRandQSet(_DepthOrderRandQSet):
    NAME = "depth-order-highestc-rand"
    CLARITY = Q_.CLARITY_HIGHEST

    TARGET_REFS = [
        "the object marked with a red circle",
    ]
    DISTRACTORS_REFS = [
        "the objects marked with a blue square",
    ]


## For in-context learning (ICL) expts
DEFAULT_N_DEMOS = 2
class BaseDemoSampler(abc.ABC):
    question_set: BaseQuestionSet
    image_set: BaseImageSet  # assuming having an info attr of type `pd.DataFrame`

    CUE_MATCHING_FUNC = all

    """Sample a few demos for ICL or few-shot for depth ordering"""
    def __init__(self, question_set, image_set):
        self.question_set = question_set
        self.image_set = image_set

    @abc.abstractmethod
    def sample(self, img_info: dict, n=DEFAULT_N_DEMOS, cot=False) -> typing.List[IclDemo]:
        """Sample demos, supporting chain of thoughts (cot).

        `img_info` dict keys:
        image_name	env_cat	env_id	obj_cat	obj_id	odd_position	cues
        """
        pass


class DemoSampler(BaseDemoSampler):
    def __init__(self, question_set, image_set):
        if not isinstance(question_set, _RandTemplateQuestionSet):
            raise TypeError("DemoSampler only supports _RandTemplateQuestionSet")

        super().__init__(question_set, image_set)

    def sample(self, img_info: dict, n=DEFAULT_N_DEMOS, cot=False) -> typing.List[IclDemo]:
        """Following best practices of sampling demos for ICL.

        1. similar images
        2. balanced numbers of pos and neg demos (depth ordering only for now)
        3. randomized order of pos and neg demos
        """
        simlar_image_info_list = self._find_similar_images(img_info, n)
        demos = []
        for sim_img_info in simlar_image_info_list:
            prompt_input = self._format_prompt()
            response_output = self._format_response(sim_img_info, prompt_input, cot)

            image = self.image_set.get_image(sim_img_info['image_name'])
            demos.append(
                IclDemo(image=image, text_in=prompt_input, text_out=response_output))

        random.shuffle(demos)
        lgr.info("sampled demos for img %s: %s",
                 img_info['image_name'], [d.image.path.split('/')[-1] for d in demos])
        return demos

    def _find_similar_images(self, img_info: dict, n: int) -> typing.List[dict]:
        """List similar (ie cue matching) image with two balanced constraints."""
        assert n % 2 == 0, "# of demos should be an even number"
        n_pos, n_neg = n // 2, n // 2

        # list of imggen.main.CUE_*
        if isinstance(img_info["cues"], str):
            cues = list(eval(img_info["cues"]))
        else:
            cues = img_info["cues"]
        assert isinstance(cues, list), "Failure to extract a list of cues"

        odd_position = img_info["odd_position"]  # 'far' or 'near'
        neg_position = 'far' if odd_position == 'near' else 'near'

        pos_info_list = self._find_images_by_attrs(cues, odd_position, n_pos,
                                                   excl_names={img_info['image_name']})
        neg_info_list = self._find_images_by_attrs(cues, neg_position, n_neg)

        return pos_info_list + neg_info_list

    def _format_prompt(self):
        return self.question_set.list_questions()[0].text

    def _format_response(self, sim_img_info, ques_text, cot):
        response = ""

        if cot:  # only help with depth ordering, not referral comprehension
            parsed_cues = eval(sim_img_info["cues"])
            cue1 = parsed_cues[0]
            odd_position = sim_img_info['odd_position']

            visual_description = self._get_obj_visual_description(cue1, odd_position)
            cue_names = self._get_cue_names(sim_img_info)
            depth_order = self._get_depth_ordering(sim_img_info)

            # support 2 cues at most
            visual_description2 = ""
            if len(parsed_cues) > 1:
                cue2 = parsed_cues[1]
                visual_description2 = self._get_obj_visual_description(cue2, odd_position)

            response += " ".join([
                "(Let's think step by step.",
                f"The object of interest {visual_description}.",
                f"Additionally, the object {visual_description2}." if visual_description2 else "",
                f"Based on the {cue_names} pictorial cues,",
                f"it is likely that the object is {depth_order} the other objects.) ",
            ])

        answer_letter = get_gt_letter(ques_text, sim_img_info['odd_position'])
        response += f'{answer_letter}.'

        return response

    def _find_images_by_attrs(self, cues, position, n, excl_names=None) -> typing.List[dict]:
        excl_names = excl_names or {}
        excluded_names = lambda df: ~df.image_name.isin(excl_names)

        cond = self.CUE_MATCHING_FUNC  # default `all`
        matching_cues = lambda df: df.cues.map(eval).apply(lambda l: cond(c in l for c in cues))

        matching_position = lambda df: df.odd_position == position
        matched = self.image_set.info.loc[matching_cues].loc[matching_position].loc[excluded_names]

        return matched.sample(n).to_dict(orient='records')

    @staticmethod
    def _get_obj_visual_description(single_cue, odd_position) -> str:
        template_map = { # short cue name -> vis description template
            "HP": "appears {detail} than the others",
            "LS": "{detail}",
            "OC": "is {detail} the other objects",
            "TG": "has {detail} texture density than the others",
            "RS": "looks {detail} than the rest",
            "FS": "is usually {detail} than the other objects",
            "FO": "is {detail} focused than the others",  # as a misleading CoT expt
            "SA": "is {detail} saturated than the rest",
            "LP": "is {detail} situated on a ground with some regular patterns",
        }

        if odd_position == 'near':
            vis_detail_map = { # short cue name -> detailed diff b/w targ & distractors
                "HP": "lower",
                "LS": "does not have any shadow casted on its front face",
                "OC": "occluding",
                "TG": "lower",
                "RS": "larger",
                "FS": "smaller",
                "FO": "more",  # as a misleading CoT expt
                "SA": "more",
                "LP": "",
            }
        else:
            vis_detail_map = {
                "HP": "higher",
                "LS": "has some shadow casted potentially by the other objects",
                "OC": "occluded by",
                "TG": "higher",
                "RS": "smaller",
                "FS": "larger",
                "FO": "less",  # as a misleading CoT expt
                "SA": "less",
                "LP": "",
            }

        template = template_map[single_cue]
        vis_detail = vis_detail_map[single_cue]
        return template.format(detail=vis_detail)

    @staticmethod
    def _get_cue_names(sim_img_info) -> str:
        cue_name_map = {  # short cue name -> readable cue name
            "HP": "height-in-plane",
            "LS": "shadow",
            "OC": "interposition",
            "TG": "texture density",
            "RS": "relative size",
            "FS": "familiar size",
            "FO": "focus",
            "SA": "aerial perspective",
            "LP": "linear perspective",
        }
        if isinstance(sim_img_info["cues"], str):
            cues = eval(sim_img_info["cues"])
        else:
            cues = sim_img_info["cues"]
        return " and ".join(cue_name_map[c] for c in cues)

    @staticmethod
    def _get_depth_ordering(sim_img_info) -> str:
        if sim_img_info['odd_position'] == 'near':
            return "closer than"
        else:
            return "farther away than"


class PermissiveDemoSampler(DemoSampler):
    CUE_MATCHING_FUNC = any


## Helper functions
def _rand_mcq(options, symbols="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Randomized options for multiple choice questions."""
    assert len(options) <= len(symbols), "Too many options"
    random.shuffle(options)
    formatted = ""
    for symbol, option in zip(symbols, options):
        formatted += f"{symbol}. {option}. "
    return formatted


OPTION_REGEX = r"(?:\?|\.) ([A-Z]\. [^\.]+\. )([A-Z]\. [^\.]+\. )([A-Z]\. [^\.]+\. )?([A-Z]\. [^\.]+\. )?"
ODD_POSITION_MAP = {  # answer variation -> 'near' or 'far
    'closer': 'near',
    'in front of': 'near',
    'before': 'near',
    'nearer': 'near',

    'farther': 'far',
    'behind': 'far',
    'after': 'far',
    'farther': 'far',
}
def get_gt_letter(ques_text, odd_position):
    matched = re.search(OPTION_REGEX, ques_text)

    # build letter_map
    letter_map = {}  # letter -> answer
    for i in range(1, 5):
        option = matched.group(i)
        if option is not None:
            letter = option[0]
            answer = option.split(" ", maxsplit=1)[1].strip(" .")
            letter_map[answer] = letter

    # figure out the correct option letter
    for answer, letter in letter_map.items():
        if ODD_POSITION_MAP.get(answer.lower()) == odd_position:
            return letter

    # sync'ed with `_DepthOrderRandQSet.TEMPLATES`
    if 'at the rear of' in ques_text:
        gt_answer = 'Yes' if odd_position == 'far' else 'No'
        return letter_map[gt_answer]
    if 'in front of' in ques_text:
        gt_answer = 'True' if odd_position == 'near' else 'False'
        return letter_map[gt_answer]

    raise ValueError(f"Can't find [{odd_position}] GT for ques: {ques_text}")


def get_question_set(name: str) -> BaseQuestionSet:
    try:
        class_ = _get_class_by_name(name)
        lgr.info("Got question set: %s", name)
        return class_()
    except KeyError:
        raise ValueError(f"Invalid question set name: {name}")


def get_image_set(name: str, **kwargs) -> BaseImageSet:
    try:
        class_ = _get_class_by_name(name)
        image_set: BaseImageSet = class_(**kwargs)
        lgr.info("Got image set '%s' with %d images", name, len(image_set))
        return image_set
    except KeyError:
        raise ValueError(f"Invalid image set name: {name}")


def get_demo_sampler(question_set, image_set) -> BaseDemoSampler:
    to_be_permissive = any((  # not enough images for `DemoSampler`
        isinstance(image_set, O3DepthOrderImageSet),
        isinstance(image_set, O3DepthRealImageSet),
        isinstance(image_set, O3DepthCFRealImageSet),
        isinstance(image_set, HfReal012CueImageSet),
        isinstance(image_set, HfReal012CueCroppedAugmentedImageSet),
        isinstance(image_set, HfReal012CueCroppedMarkedImageSet),
        isinstance(image_set, HfRealMixedCueImageSet),
    ))
    if to_be_permissive:
        return PermissiveDemoSampler(question_set, image_set)
    else:
        return DemoSampler(question_set, image_set)


def get_image_info(name, img_set: BaseImageSet) -> dict:
    try:
        lgr.debug("getting image info for %s", name)
        name = re.sub(util.AUG_SUFFIX_REGEX, r"", name)
        info_df = img_set.info.loc[lambda df: df.image_name == name]
        assert info_df.shape[0] == 1, f"Can't locate unique image by fname: {name}"
        image_info = info_df.iloc[0].to_dict()
        return image_info

    except AttributeError:
        raise ValueError(f"ImageSet {img_set} does not have a info df.")


def _get_class_by_name(name: str) -> type:
    import importlib
    expt_mod = importlib.import_module("expt")

    all_classes = []
    for d in dir(expt_mod):
        attr = getattr(expt_mod, d)
        if isinstance(attr, type):
            all_classes.append(attr)

    class_map = {  # name -> class
        c.NAME: c
        for c in all_classes if hasattr(c, "NAME")
    }
    return class_map[name]


def prep_question_kwargs(question_set, image_set, image_name) -> dict:
    if isinstance(question_set, HfVisualQuestionSet):
        return HfVisualQuestionSet.prep_question_kwargs(image_set, image_name)
    return {}
