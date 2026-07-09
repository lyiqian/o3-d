"""Kosmos"""

import dataclasses
import logging
import typing
import torch
from tqdm import tqdm

import torch

from transformers import AutoProcessor, AutoModelForVision2Seq

import core
import expt
import util


lgr = logging


class KsImagePreprocessor(core.IImagePreprocessor):
    def preprocess(self, img: core.FileSysImage):
        return img.to_pil()


class KsPromptFormatter(core.IPromptFormatter):
    def format(self, ques: core.Question) -> str:
        # kosmos not happy with "Answer A or B"
        return f"<grounding> Question: {ques.text.replace('Answer A or B.', '')} Answer:"


class KosmosVLM(core.BaseVLM):

    def init_model(self):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.processor = AutoProcessor.from_pretrained(
            f'microsoft/{self.model_name}', device_map='auto')

        self.model = AutoModelForVision2Seq.from_pretrained(
            f'microsoft/{self.model_name}',
            device_map='auto',
            cache_dir=util.HF_CACHE).to(self.dev)

    def _vqa(self, prompt, img_) -> core.BaseResponse:
        # few-shot: probably unsupported
        # https://github.com/microsoft/unilm/issues/1451
        lgr.debug(prompt)
        processed_inputs = self.processor(text=prompt, images=img_, return_tensors='pt').to(self.dev)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=processed_inputs["pixel_values"],
                input_ids=processed_inputs["input_ids"],
                attention_mask=processed_inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=processed_inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=128,
            )
        return core.GenerationResp(content=generated_ids)

    def _vqa_icl(self, prompt, img, demos):
        raise NotImplementedError("skipped for now")


class HfProcessorTextExtractor(core.BaseExtractor):

    def __init__(self, processor):
        self.processor = processor

    def extract(self, resp: core.GenerationResp) -> core.TextResult:
        generated_text = self.processor.batch_decode(resp.content, skip_special_tokens=True)[0]
        processed_text, entities = self.processor.post_process_generation(generated_text)
        extracted = processed_text.split("Answer:")[1].strip() if "Answer:" in processed_text else processed_text
        return core.TextResult(text=extracted)


def main():
    args = util.get_argparser().parse_args()

    image_set = expt.get_image_set(args.image_set, **args.imset_kwargs)
    image_names = image_set.list_image_names()
    question_set = expt.get_question_set(args.question_set)

    lgr.info('Loading model: %s', args.model_name)

    ks_vlm = KosmosVLM(
        model_name=args.model_name,
        formatter=KsPromptFormatter(),
        image_preprocessor=KsImagePreprocessor())

    text_extractor = HfProcessorTextExtractor(processor=ks_vlm.processor)

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
                resp = ks_vlm.vqa(ques=ques, img=image)
                answer = text_extractor.extract(resp)
                answer.text = answer.text.replace(ques.text, "").strip()
                lgr.debug(answer)
                record.update(**answer.to_record())

                result_records.append(record)

        util.save_results(args, result_records, ts_str=timestamp)

    lgr.info("Finished %d batches of images with %s", n, args.model_name)


if __name__ == '__main__':
    main()


"""
# model_names
# kosmos-2-patch14-224
# https://huggingface.co/microsoft/kosmos-2-patch14-224

DEBUG=1 python3 main.py --model_name kosmos-2-patch14-224
"""
