"""Template/starter/to-copy-paste"""

import dataclasses
import logging
import typing
import torch
from tqdm import tqdm

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.mm_utils import tokenizer_image_token, process_images

import core
import expt
import util


lgr = logging


class CbImagePreprocessor(core.IImagePreprocessor):
    def preprocess(self, img: core.FileSysImage):
        return img.to_pil()


class CbPromptFormatter(core.IPromptFormatter):
    def format(self, ques: core.Question) -> str:
        return ques.text


class CambrianVLM(core.BaseVLM):
    """Based on https://github.com/cambrian-mllm/cambrian/blob/main/inference.py"""

    CONV_MODE_MAP = {
        # https://github.com/cambrian-mllm/cambrian/blob/539ffc3254bba004e5d012b65c0ad6cb308897c5/cambrian/conversation.py#L552
        "cambrian-phi3-3b": "phi3",
        "cambrian-8b": "llama_3"
    }

    TEMPERATURE = 0

    def init_model(self):
        self.tokenizer, self.model, self.image_processor, __ = load_pretrained_model(
            f'nyu-visionx/{self.model_name}', None, self.model_name,
            load_4bit=True,
            device_map='auto', cache_dir=util.HF_CACHE,
        )

    def _vqa(self, prompt, img_) -> core.BaseResponse:
        lgr.debug(prompt)
        input_ids, image_tensor, image_sizes, prompt = self._process(img_, prompt)
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,  # sampling done via asking same Q for diff images
                num_beams=1,
                max_new_tokens=512,
                use_cache=True)

        return core.GenerationResp(content=output_ids)

    def _vqa_icl(self, prompt, img, demos):
        # https://github.com/cambrian-mllm/cambrian/issues/4#issuecomment-2197442909
        raise NotImplementedError("Cambrian does not support ICL")

    def _process(self, image, question):
        qs = question

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv_mode = self.CONV_MODE_MAP[self.model_name]
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)  # "\n<|user|>\n"
        conv.append_message(conv.roles[1], None)  # "\n<|assistant|>\n"
        prompt = conv.get_prompt()

        image_size = [image.size]
        image_tensor = process_images([image], self.image_processor, self.model.config)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        return input_ids, image_tensor, image_size, prompt


class TokenizerTextExtractor(core.BaseExtractor):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract(self, resp: core.GenerationResp) -> core.TextResult:
        text = self.tokenizer.batch_decode(resp.content, skip_special_tokens=True)[0].strip()
        return core.TextResult(text=text)



def main():
    args = util.get_argparser().parse_args()

    # Getting images and questions before the slow model init
    image_set = expt.get_image_set(args.image_set, **args.imset_kwargs)
    image_names = image_set.list_image_names()
    question_set = expt.get_question_set(args.question_set)

    lgr.info('Loading model: %s', args.model_name)

    cb_vlm = CambrianVLM(
        model_name=args.model_name,
        formatter=CbPromptFormatter(),
        image_preprocessor=CbImagePreprocessor())

    text_extractor = TokenizerTextExtractor(tokenizer=cb_vlm.tokenizer)

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
                resp = cb_vlm.vqa(ques=ques, img=image)
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
# cambrian-phi3-3b
# https://huggingface.co/nyu-visionx/cambrian-phi3-3b
# cambrian-8b  # OOM @ 4bit on 1080 Ti

DEBUG=1 python3 main.py --model_name cambrian-phi3-3b
"""
