"""Template/starter/to-copy-paste"""

import dataclasses
import logging
import typing
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

import core
import expt
import util


lgr = logging

class PhiImagePreprocessor(core.IImagePreprocessor):
    def preprocess(self, img: core.FileSysImage):
        return img.to_pil()


class PhiPromptFormatter(core.IPromptFormatter):
    def format(self, ques: core.Question) -> str:
        return ques.text

@dataclasses.dataclass
class PhiGenerationResp(core.GenerationResp):
    input_len: int


class PhiVLM(core.BaseVLM):

    def init_model(self):
        self.processor = AutoProcessor.from_pretrained(
            f'microsoft/{self.model_name}',
            trust_remote_code=True,
            num_crops=4)  # rec'ed 16 caused CUDA OOM

        self.model = AutoModelForCausalLM.from_pretrained(
            f'microsoft/{self.model_name}',
            device_map='auto',
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation='eager',
            cache_dir=util.HF_CACHE)

    def _vqa(self, prompt, img_) -> core.BaseResponse:
        lgr.debug(prompt)

        images = [img_]
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"},
        ]
        response = self._generate(images, messages)
        return response

    def _vqa_icl(self, prompt, img_, demos):
        # https://huggingface.co/microsoft/Phi-3.5-vision-instruct#input-formats
        lgr.debug(prompt)

        images, messages = [], []
        img_idx = 1

        for demo in demos:
            images.append(demo.image.to_pil())
            messages.extend([
                {"role": "user", "content": f"<|image_{img_idx}|>\n{demo.text_in}"},
                {"role": "assistant", "content": demo.text_out},
            ])
            img_idx += 1

        images.append(img_)
        messages.append(
            {"role": "user", "content": f"<|image_{img_idx}|>\n{prompt}"}
        )

        response = self._generate(images, messages)
        return response

    def _generate(self, images, messages):
        prompt_ = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            prompt_, images, return_tensors="pt").to(self.model.device)

        generation_args = {
            "max_new_tokens": 1000,
            # "temperature": 0.0,  # unset when do_sample=False, according to warning msg
            "do_sample": False,
        }

        generate_ids = self.model.generate(**inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args
        )

        return PhiGenerationResp(content=generate_ids, input_len=inputs['input_ids'].shape[1])



class HfProcessorTextExtractor(core.BaseExtractor):

    def __init__(self, processor):
        self.processor = processor

    def extract(self, resp: PhiGenerationResp) -> core.TextResult:
        generate_ids = resp.content
        truncated = generate_ids[:, resp.input_len:]

        text = self.processor.batch_decode(truncated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]

        return core.TextResult(text=text)



def main():
    args = util.get_argparser().parse_args()

    image_set = expt.get_image_set(args.image_set, **args.imset_kwargs)
    image_names = image_set.list_image_names()
    question_set = expt.get_question_set(args.question_set)

    lgr.info('Loading model: %s', args.model_name)

    phi_vlm = PhiVLM(
        model_name=args.model_name,
        formatter=PhiPromptFormatter(),
        image_preprocessor=PhiImagePreprocessor())

    text_extractor = HfProcessorTextExtractor(processor=phi_vlm.processor)

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
                    resp = phi_vlm.vqa_icl(ques=ques, img=image, demos=demos)
                    record.update(demos=[str(d) for d in demos])
                else:
                    resp = phi_vlm.vqa(ques=ques, img=image)
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
# Phi-3.5-vision-instruct
# https://huggingface.co/microsoft/Phi-3.5-vision-instruct

DEBUG=1 python3 main.py --model_name Phi-3.5-vision-instruct
"""
