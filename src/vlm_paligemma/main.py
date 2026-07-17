import dataclasses
import logging
import re
import typing

from huggingface_hub import login
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from tqdm import tqdm
import torch

import core
import expt
import util


lgr = logging


class PgPromptFormatter(core.IPromptFormatter):

    def format(self, ques: core.Question) -> str:
        # Prompt formatting as explained in
        # https://github.com/huggingface/blog/blob/main/paligemma.md
        # add <image> for each image in the prompt
        # add beginning of sentence <bos> token after the images
        # add newline after the prompt
        return f'<image><bos>{ques.text}\n'


class PgImagePreprocessor(core.IImagePreprocessor):
    def preprocess(self, img: core._RegularImage):
        return img.to_pil().convert("RGB")


@dataclasses.dataclass
class PgGenerationResp(core.GenerationResp):
    input_len: int


class PaligemmaVLM(core.BaseVLM):

    def init_model(self):
        torch.manual_seed(1234)
        login(token=util.HF_TOKEN)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_name, token=util.HF_TOKEN, device_map="auto",
            cache_dir=util.HF_CACHE)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, token=util.HF_TOKEN)

    def vqa(self, ques: core.Question, img: core.BaseImage) -> core.BaseResponse:
        prompt = self.prep_prompt(ques)
        img_ = self.prep_img(img)
        return self._vqa(prompt, img_)

    def _vqa(self, prompt, img_) -> core.BaseResponse:
        lgr.debug("Prepared prompt is: %s", prompt)
        model_inputs = self.processor(text=prompt, images=img_, return_tensors='pt').to(self.device)

        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)

        input_len = model_inputs['input_ids'].shape[-1]

        return PgGenerationResp(content=generation, input_len=input_len)

    def _vqa_icl(self, prompt, img, demos):
        raise NotImplementedError("skipped for now")


class HfProcessorTextExtractor(core.BaseExtractor):

    def __init__(self, processor):
        self.processor = processor

    def extract(self, resp: PgGenerationResp) -> core.TextResult:
        content = resp.content[0][resp.input_len:]
        raw_result = self.processor.decode(content, skip_special_tokens=True)
        return core.TextResult(text=raw_result)


def main():
    args = util.get_argparser().parse_args()


    image_set = expt.get_image_set(args.image_set, **args.imset_kwargs)
    image_names = image_set.list_image_names()

    question_set = expt.get_question_set(args.question_set)

    lgr.info('Loading model: %s', args.model_name)
    pg_vlm = PaligemmaVLM(
        model_name=args.model_name,
        formatter=PgPromptFormatter(),
        image_preprocessor=PgImagePreprocessor())

    text_extractor = HfProcessorTextExtractor(processor=pg_vlm.processor)

    timestamp, result_records = util.prepare_vlm_run(args)

    batches_of_names = util.batches_of(image_names, size=100)
    for n, names in enumerate(batches_of_names, start=1):
        for image_fname in tqdm(names, desc=f"Batch {n}"):
            kwargs = expt.prep_question_kwargs(question_set, image_set, image_fname)
            questions = question_set.list_questions(**kwargs)

            if util.to_skip_image(args, image_fname, len(questions), result_records):
                continue

            image = image_set.get_image(image_fname)
            for ques in questions:
                record = {
                    "image_name": image_fname,
                    **ques.to_record(),
                }
                lgr.info(record)

                if args.icl:
                    raise ValueError(f"{args.model_name} does not support ICL")
                resp = pg_vlm.vqa(ques=ques, img=image)
                answer = text_extractor.extract(resp)
                lgr.debug(answer)
                record.update(**answer.to_record())

                result_records.append(record)

        util.save_results(args, result_records, ts_str=timestamp)

    lgr.info("Finished %d batches of images with %s", n, args.model_name)

if __name__ == "__main__":
    main()

"""
# https://huggingface.co/google/paligemma2-3b-mix-224

# PaliGemma 2 is not a multi-turn chatbot. It is designed for a single round of image and text input.

DEBUG=1 python3 main.py --model_name google/paligemma2-3b-mix-224
"""