import abc
import dataclasses
import logging as lgr
import io
import pathlib
import re
import typing

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np

import util


@dataclasses.dataclass
class Question:
    CLARITY_LOWEST = 1
    CLARITY_LOW = 2
    CLARITY_MEDIUM = 3
    CLARITY_HIGH = 3.5
    CLARITY_HIGHEST = 4

    text: str
    kind: str = "qualitative"
    clarity: int = CLARITY_LOW

    def __str__(self):
        return self.text

    def to_record(self) -> dict:
        return {
            "ques": self.text,
            "ques_kind": self.kind,
            "ques_clarity": self.clarity
        }

    def format(self, **kwargs) -> "Question":
        return self.__class__(text=self.text.format(**kwargs), kind=self.kind, clarity=self.clarity)


@dataclasses.dataclass
class BaseImage:
    path: str


@dataclasses.dataclass
class IclDemo:
    image: BaseImage
    text_in: str
    text_out: str

def __str__(self):
    return f"Img: {self.image.path}; Text-in: {self.text_in}; Text-out: {self.text_out}."


@dataclasses.dataclass
class BaseResponse:
    content: typing.Any

    def __str__(self):
        return str(self.content)


class IPromptFormatter(abc.ABC):
    @abc.abstractmethod
    def format(self, ques: Question) -> str:
        pass


class IImagePreprocessor(abc.ABC):
    @abc.abstractmethod
    def preprocess(self, img: BaseImage):
        pass


class BaseVLM(abc.ABC):
    model_name: str
    formatter: IPromptFormatter
    image_preprocessor: IImagePreprocessor

    def __init__(self, model_name, formatter, image_preprocessor):
        self.model_name = model_name
        self.formatter = formatter
        self.image_preprocessor = image_preprocessor

        self.init_model()

    @abc.abstractmethod
    def init_model(self):
        pass

    @abc.abstractmethod
    def _vqa(self, prompt, img_) -> BaseResponse:
        pass

    @abc.abstractmethod
    def _vqa_icl(self, prompt, img_, demos) -> BaseResponse:
        pass

    def prep_prompt(self, ques: Question) -> str:
        return self.formatter.format(ques)

    def prep_img(self, img: BaseImage):
        return self.image_preprocessor.preprocess(img)

    def vqa(self, ques: Question, img: BaseImage) -> BaseResponse:
        prompt = self.prep_prompt(ques)
        img_ = self.prep_img(img)
        return self._vqa(prompt, img_)

    def vqa_icl(self, ques: Question, img: BaseImage, demos: typing.List[IclDemo]) -> BaseResponse:
        question = self.prep_prompt(ques)
        img_ = self.prep_img(img)
        return self._vqa_icl(question, img_, demos)


class BaseResult(abc.ABC):
    @abc.abstractmethod
    def to_record(self):
        pass


class BaseExtractor(abc.ABC):
    @abc.abstractmethod
    def extract(self, response: BaseResponse) -> BaseResult:
        pass


@dataclasses.dataclass
class BaseGroundTruth(abc.ABC):
    src: str

    @abc.abstractmethod
    def load(self):
        pass


class BaseEvaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, result: BaseResult, gt: BaseGroundTruth):
        pass


## Abstraction lvl 2
@dataclasses.dataclass
class _RegularImage(BaseImage):
    image_set_name: str  # (path.name, image_set_name) can be used as unique ID

    @abc.abstractmethod
    def to_pil(self):
        pass

    @abc.abstractmethod
    def to_filesys_path(self, **kwargs):
        pass

    @abc.abstractmethod
    def to_bytes(self):
        pass

    def format_unique_id(self):
        return f"{self.image_set_name}_{pathlib.Path(self.path).name}"


class FileSysImage(_RegularImage):
    def to_pil(self):
        lgr.debug('Reading img: %s', self.path)
        img = PIL.Image.open(self.path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def to_numpy(self):
        lgr.debug('Reading img: %s', self.path)
        return plt.imread(self.path)

    def to_filesys_path(self, **kwargs):
        return self.path

    def to_bytes(self):
        return open(self.path, "rb")

    def format_unique_id(self):
        # best effort: relying on dataaug.py creating unique names on file system
        return pathlib.Path(self.path).name


class HuggingFaceImage(_RegularImage):
    """Awkward, but as an adaptor for pre-huggingface experiment code"""
    def __init__(self, *args, hf_dataset=None, img_col='image'):
        super().__init__(*args)

        self.hf_dataset = hf_dataset
        self.img_col = img_col
        self._sample = None

    def to_pil(self):
        if self._sample is None:
            self._set_sample()

        return self._sample[self.img_col]

    def to_filesys_path(self, suffix=""):
        if self._sample is None:
            self._set_sample()

        tmp_path = f".tmp{suffix}.png"
        self._sample[self.img_col].save(tmp_path)
        return tmp_path

    def to_bytes(self):
        raise NotImplementedError

    def _set_sample(self):
        self._sample = util.df_dataset_get_sample(self.hf_dataset, self.path)

class HuggingFaceBytesImage(HuggingFaceImage):

    def to_pil(self):
        raise NotImplementedError

    def to_filesys_path(self, suffix=""):
        tmp_path = f".tmp{suffix}.png"
        with open(tmp_path, "wb") as fo:
            fo.write(self.to_bytes().read())
        return tmp_path

    def to_bytes(self):
        if self._sample is None:
            self._set_sample()

        image_content = io.BytesIO(self._sample[self.img_col]['bytes'])
        image_content.name = self.format_unique_id()  # for OpenAI API upload
        return image_content


class NoopFormatter(IPromptFormatter):
    def format(self, ques: Question) -> str:
        assert isinstance(ques, Question)
        return ques.text


class NoopImagePreprocessor(IImagePreprocessor):
    def preprocess(self, img: BaseImage) -> BaseImage:
        return img


@dataclasses.dataclass
class GenerationResp(BaseResponse):
    pass


@dataclasses.dataclass
class TextResult(BaseResult):
    text: str
    truncated: str = False

    def to_record(self):
        return {'answer': self.text, 'answer_truncated': self.truncated}

    @classmethod
    def from_record(cls, record: dict) -> 'TextResult':
        return cls(text=record['answer'],
                   truncated=record.get('answer_truncated', False))


@dataclasses.dataclass
class ErrorResult(BaseResult):
    error: str

    def to_record(self):
        return {'error': self.error}


@dataclasses.dataclass
class BinaryGroundTruth(BaseGroundTruth):

    def load(self) -> bool:
        return bool(self.src)


@dataclasses.dataclass
class TextGroundTruth(BaseGroundTruth):
    def load(self) -> str:
        return str(self.src)


@dataclasses.dataclass
class DataframeRowGT(BaseGroundTruth):
    def load(self) -> dict:
        return self.src.to_dict()


@dataclasses.dataclass
class OrigImgGT(BaseGroundTruth):
    def load(self):
        img = plt.imread(self.src)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return img


class CloserFartherBinaryEvaluator(BaseEvaluator):
    def evaluate(self, result: TextResult, gt: TextGroundTruth):
        actual_answer = result.text.lower()
        expected_answer = gt.load()
        return float(expected_answer in actual_answer)


class CloserFartherFilenameEvaluator(BaseEvaluator):
    EXPECTED_NEAR_ANSWERS = {
        'closer',
        'before', 'nearer', 'in front of',
    }
    EXPECTED_FAR_ANSWERS = {
        'farther', 'further',
        'behind', 'after'
    }
    def evaluate(self, result: TextResult, gt: DataframeRowGT):
        actual_answer = result.text.lower()

        gt_dict = gt.load()
        image_name = gt_dict['image_name']
        if '_near' in image_name:
            expected_answers = self.EXPECTED_NEAR_ANSWERS
        elif '_far' in image_name:
            expected_answers = self.EXPECTED_FAR_ANSWERS
        else:
            # raise ValueError(f"Unexpected image_name: {image_name}")
            return None  # for '_none_'

        return any(expected in actual_answer for expected in expected_answers)


class CloserFartherOddPosEvaluator(BaseEvaluator):
    def evaluate(self, result: TextResult, gt: DataframeRowGT):
        actual_answer = result.text.lower()

        gt_dict = gt.load()
        odd_pos = gt_dict['odd_position']
        if odd_pos == 'near':
            expected_answers = ['closer']
        elif odd_pos == 'far':
            expected_answers = ['farther', 'further']
        else:
            return None  # for 'none'

        return any(expected in actual_answer for expected in expected_answers)


class CloserFartherO3dcfrEvaluator(BaseEvaluator):
    def evaluate(self, result: TextResult, gt: DataframeRowGT):
        actual_answer = result.text.lower()

        gt_dict = gt.load()
        gt_pos = self._get_gt_pos(gt_dict['qtags'], gt_dict['full_image_name'])

        if gt_pos == 'near':
            expected_answers = ['closer']
        elif gt_pos == 'far':
            expected_answers = ['farther', 'further']
        else:
            return None  # for 'none'

        return any(expected in actual_answer for expected in expected_answers)

    def _get_gt_pos(self, qtags, full_image_name):
        gt_str = re.search(r"-t([lr](far|near))", full_image_name)[1]
        gt_lr = gt_str[0]
        gt_pos = gt_str[1:]

        if '35l' in qtags or '35r' in qtags:
            adjusted_gt_pos_map = {
                ('35l', 'l', 'far'): 'far',
                ('35l', 'l', 'near'): 'near',
                ('35l', 'r', 'far'): 'near',
                ('35l', 'r', 'near'): 'far',
                ('35r', 'l', 'far'): 'near',
                ('35r', 'l', 'near'): 'far',
                ('35r', 'r', 'far'): 'far',
                ('35r', 'r', 'near'): 'near',
            }
            for (tag, lr, pos), adjusted_gt_pos in adjusted_gt_pos_map.items():
                if tag in qtags and lr == gt_lr and pos == gt_pos:
                    return adjusted_gt_pos
            raise RuntimeError("MY BUG")

        else: # mark
            return gt_pos
