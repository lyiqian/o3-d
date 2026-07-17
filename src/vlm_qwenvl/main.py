import dataclasses
import logging
import json
import typing

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
import torch

import core
import expt
import util

lgr = logging

QwImagePreprocessor = core.NoopImagePreprocessor


class QwPromptFormatter(core.IPromptFormatter):
    def format(self, ques: core.Question) -> str:
        return ques.text


@dataclasses.dataclass
class QwGenerationResp(core.GenerationResp):
    input_ids: typing.List[int]


class QwenVLM(core.BaseVLM):

    MIN_PIXELS = 256 * 28 * 28
    MAX_PIXELS = 1280 * 28 * 28
    MAX_NEW_TOKENS = 1024 * 1

    def init_model(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            f'Qwen/{self.model_name}', torch_dtype="auto", device_map="auto",
            cache_dir=util.HF_CACHE)

        self.processor = AutoProcessor.from_pretrained(
            f'Qwen/{self.model_name}',
            min_pixels=self.MIN_PIXELS, max_pixels=self.MAX_PIXELS)

    def _vqa(self, prompt, img_) -> core.BaseResponse:
        lgr.debug(prompt)
        conversations = self._format_conversations(prompt, img_)
        response = self._generate(conversations)
        return response

    def _vqa_icl(self, prompt, img_, demos):
        lgr.debug(prompt)
        conversations = self._format_conversations_icl(prompt, img_, demos)
        response = self._generate(conversations)
        return response

    def _format_conversations(self, prompt, img_: core._RegularImage):
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f'file://{img_.to_filesys_path()}'},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return conversations

    def _format_conversations_icl(self, prompt, img_: core._RegularImage, demos):
        conversations = []
        for i, demo in enumerate(demos):
            conversations.extend([
                {"role": "user",
                 "content": [
                    # using a suffix so tmp files are not overwritten each other before query
                    {"type": "image", "image": f'file://{demo.image.to_filesys_path(suffix=i)}'},
                    {"type": "text", "text": demo.text_in}]
                },
                {"role": "assistant",
                 "content": demo.text_out}
            ])
        conversations.extend([
            {"role": "user",
             "content": [
                 {"type": "image", "image": f'file://{img_.to_filesys_path()}'},
                 {"type": "text", "text": prompt}]
            }
        ])
        return conversations

    def _generate(self, conversations):
        text = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        with torch.no_grad():
            image_inputs, video_inputs = process_vision_info(conversations)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=self.MAX_NEW_TOKENS)

        return QwGenerationResp(content=generated_ids, input_ids=inputs.input_ids)

class HfProcessorTextExtractor(core.BaseExtractor):

    def __init__(self, processor):
        self.processor = processor

    def extract(self, resp: QwGenerationResp) -> core.TextResult:
        trimmed_gen_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(resp.input_ids, resp.content)
        ]
        raw_result = self.processor.batch_decode(
            trimmed_gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        text = raw_result[0]
        return core.TextResult(text=text)


def main():
    args = util.get_argparser().parse_args()

    image_set = expt.get_image_set(args.image_set, **args.imset_kwargs)
    image_names = image_set.list_image_names()
    question_set = expt.get_question_set(args.question_set)

    lgr.info('Loading model: %s', args.model_name)
    qw_vlm = QwenVLM(
        model_name=args.model_name,
        formatter=QwPromptFormatter(),
        image_preprocessor=QwImagePreprocessor())
    text_extractor = HfProcessorTextExtractor(processor=qw_vlm.processor)

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
                    "icl": args.icl, "cot": args.cot,
                    **ques.to_record()
                }
                lgr.info(record)

                if args.icl:
                    demo_sampler = expt.get_demo_sampler(question_set, image_set)
                    img_info = expt.get_image_info(image_fname, image_set)
                    demos = demo_sampler.sample(img_info, cot=args.cot)
                    resp = qw_vlm.vqa_icl(ques=ques, img=image, demos=demos)
                    record.update(demos=[str(d) for d in demos])
                else:
                    resp = qw_vlm.vqa(ques=ques, img=image)
                answer = text_extractor.extract(resp)
                lgr.debug(answer)
                record.update(**answer.to_record())

                result_records.append(record)

        util.save_results(args, result_records, ts_str=timestamp)

    lgr.info("Finished %d batches of images with %s", n, args.model_name)


if __name__ == "__main__":
    main()


"""
# model names
Qwen2-VL-2B-Instruct
Qwen2-VL-7B-Instruct-GPTQ-Int4
# https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4

DEBUG=1 python3 main.py --model_name Qwen2-VL-7B-Instruct-GPTQ-Int4
"""
