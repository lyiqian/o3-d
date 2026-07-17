"""LLaVa1.5"""

import dataclasses
import logging
import re
import typing
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import core
import expt
import util

IMAGE_TOKEN_SE = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN


lgr = logging


class LvImagePreprocessor(core.IImagePreprocessor):
    def preprocess(self, img: core.FileSysImage):
        return img.to_pil()


class LvPromptFormatter(core.IPromptFormatter):
    def format(self, ques: core.Question) -> str:
        return ques.text


class LlavaVLM(core.BaseVLM):

    def init_model(self):

        self.model_base = None
        self.conv_mode = None
        self.temperature = 0.2
        self.top_p = None
        self.num_beams = 1
        self.max_new_tokens = 512

        _model_path = f'liuhaotian/{self.model_name}'

        _model_name = get_model_name_from_path(_model_path)
        if "llama-2" in _model_name.lower():
            model_conv_mode = "llava_llama_2"
        elif "mistral" in _model_name.lower():
            model_conv_mode = "mistral_instruct"
            # model_conv_mode = "mistral_direct"
        elif "v1.6-34b" in _model_name.lower():
            model_conv_mode = "chatml_direct"
        elif "v1" in _model_name.lower():
            model_conv_mode = "llava_v1"
        elif "mpt" in _model_name.lower():
            model_conv_mode = "mpt"
        else:
            model_conv_mode = "llava_v0"

        if self.conv_mode is not None and self.conv_mode != model_conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    self.conv_mode, model_conv_mode, self.conv_mode
                )
            )
        else:
            self.conv_mode = model_conv_mode

        disable_torch_init()

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(_model_path, self.model_base, _model_name,
                                  load_8bit=True, device_map="auto",
                                  cache_dir=util.HF_CACHE)
        )

    def _vqa(self, prompt, img_) -> core.BaseResponse:
        # for few-shot, might need major revision; but see qwen3
        # https://huggingface.co/spaces/Qwen/Qwen3-VL-Demo
        lgr.debug(prompt)

        if IMAGE_PLACEHOLDER in prompt:
            if self.model.config.mm_use_im_start_end:
                prompt = re.sub(IMAGE_PLACEHOLDER, IMAGE_TOKEN_SE, prompt)
            else:
                prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt)
        else:
            if self.model.config.mm_use_im_start_end:
                prompt = IMAGE_TOKEN_SE + "\n" + prompt
            else:
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_ = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(prompt_, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        images = [img_]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)
        image_sizes = [x.size for x in images]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=False,  # sampling done via asking same Q for diff images
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return core.GenerationResp(content=outputs)


    def _vqa_icl(self, prompt, img, demos):
        raise NotImplementedError("skipped for now")

class BasicTextExtractor(core.BaseExtractor):
    def extract(self, resp: core.GenerationResp) -> core.TextResult:
        return core.TextResult(text=resp.content)



def main():
    args = util.get_argparser().parse_args()

    image_set = expt.get_image_set(args.image_set, **args.imset_kwargs)
    image_names = image_set.list_image_names()
    question_set = expt.get_question_set(args.question_set)

    lgr.info('Loading model: %s', args.model_name)

    llava_vlm = LlavaVLM(
        model_name=args.model_name,
        formatter=LvPromptFormatter(),
        image_preprocessor=LvImagePreprocessor())

    text_extractor = BasicTextExtractor()

    timestamp, result_records = util.prepare_vlm_run(args)

    batches_of_names = util.batches_of(image_names, size=100)
    for n, names in enumerate(batches_of_names, start=1):
        lgr.info("Processing batch %d", n)
        for image_fname in tqdm(names, desc=f"Batch {n}", total=len(names)):
            kwargs = expt.prep_question_kwargs(question_set, image_set, image_fname)
            questions = question_set.list_questions(**kwargs)

            if util.to_skip_image(args, image_fname, len(questions), result_records):
                continue

            image = image_set.get_image(image_fname)
            for ques in questions:
                record = {
                    "image_name": image_fname,
                    **ques.to_record()
                }
                lgr.info(record)

                if args.icl:
                    raise ValueError(f"{args.model_name} does not support ICL")
                resp = llava_vlm.vqa(ques=ques, img=image)
                answer = text_extractor.extract(resp)
                lgr.debug(answer)
                record.update(**answer.to_record())

                result_records.append(record)

        util.save_results(args, result_records, ts_str=timestamp)

    lgr.info("Finished %d batches of images with %s", n, args.model_name)


if __name__ == '__main__':
    main()


"""
# model_names
# llava-v1.5-7b
# https://huggingface.co/liuhaotian/llava-v1.5-7b
# llava-v1.5-7b-lora

DEBUG=1 python3 main.py --model_name llava-v1.5-7b
"""