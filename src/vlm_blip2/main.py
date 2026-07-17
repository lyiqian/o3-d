"""BLIP2"""

import dataclasses
import logging
import typing
import torch
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import core
import expt
import util


lgr = logging


class BlpImagePreprocessor(core.IImagePreprocessor):
    def preprocess(self, img: core.FileSysImage):
        return img.to_pil()


class BlpPromptFormatter(core.IPromptFormatter):
    def format(self, ques: core.Question) -> str:
        return f'Question: {ques.text} Answer:'


class Blip2VLM(core.BaseVLM):

    def init_model(self):
        self.processor = Blip2Processor.from_pretrained(
            f'Salesforce/{self.model_name}')

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            f'Salesforce/{self.model_name}',
            # load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map='auto',
            cache_dir=util.HF_CACHE)

    def _vqa(self, prompt, img_) -> core.BaseResponse:
        lgr.debug(prompt)
        inputs = self.processor(images=img_, text=prompt, return_tensors="pt")
        inputs = inputs.to(device="cuda", dtype=torch.float16)
        generated_ids = self.model.generate(**inputs)
        return core.GenerationResp(content=generated_ids)

    def _vqa_icl(self, prompt, img, demos):
        raise NotImplementedError("skipped for now")


class HfProcessorTextExtractor(core.BaseExtractor):

    def __init__(self, processor):
        self.processor = processor

    def extract(self, resp: core.GenerationResp) -> core.TextResult:
        text = self.processor.batch_decode(resp.content, skip_special_tokens=True)[0].strip()
        return core.TextResult(text=text)



def main():
    args = util.get_argparser().parse_args()

    image_set = expt.get_image_set(args.image_set, **args.imset_kwargs)
    image_names = image_set.list_image_names()
    question_set = expt.get_question_set(args.question_set)

    lgr.info('Loading model: %s', args.model_name)

    blp_vlm = Blip2VLM(
        model_name=args.model_name,
        formatter=BlpPromptFormatter(),
        image_preprocessor=BlpImagePreprocessor())

    text_extractor = HfProcessorTextExtractor(processor=blp_vlm.processor)

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
                resp = blp_vlm.vqa(ques=ques, img=image)
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
# blip2-flan-t5-xl instruction-trained
# https://huggingface.co/Salesforce/blip2-flan-t5-xl
# blip2-flan-t5-xxl  # best on VQA, instruction-trained

DEBUG=1 python3 main.py --model_name blip2-flan-t5-xl
"""
