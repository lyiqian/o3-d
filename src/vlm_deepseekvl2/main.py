import dataclasses
import logging
import typing
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

# from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
# from deepseek_vl2.utils.io import load_pil_images

import core
import expt
import util


lgr = logging

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

DsImagePreprocessor = core.NoopImagePreprocessor


class DsPromptFormatter(core.IPromptFormatter):
    def format(self, ques: core.Question) -> str:
        return ques.text


class DeepSeek2VLM(core.BaseVLM):

    def init_model(self):
        Processor = DeepseekVLV2Processor if '-vl2-' in self.model_name else VLChatProcessor
        self.processor = Processor.from_pretrained(
            f'deepseek-ai/{self.model_name}')

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            f'deepseek-ai/{self.model_name}',
            torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
            quantization_config=quantization_config,
            cache_dir=util.HF_CACHE)

    def _vqa(self, prompt, img_) -> core.BaseResponse:
        lgr.debug(prompt)
        conversations = self._get_format_conversations()(prompt, img_)
        response = self._generate(conversations)
        return response

    def _vqa_icl(self, prompt, img_, demos):
        lgr.debug(prompt)
        _format_conversation = self._get_format_conversations(icl=True)
        conversations = _format_conversation(prompt, img_, demos)
        response = self._generate(conversations)
        return response

    def _get_format_conversations(self, icl=False):
        if '-vl2-' in self.model_name:
            if icl:
                pass  # todo
            else:
                return self.format_conversations
        else:
            if icl:
                return self.format_conversations_v1_icl
            else:
                return self.format_conversations_v1

    def _generate(self, conversations) -> core.BaseResponse:
        pil_images = load_pil_images(conversations)
        processed_inputs = self.processor(
            conversations=conversations,
            images=pil_images,
            force_batchify=True
        )
        processed_inputs = processed_inputs.to(self.model.device)

        with torch.no_grad():
            # run image encoder to get the image embeddings
            inputs_embeds = self.model.prepare_inputs_embeds(**processed_inputs)

            with torch.inference_mode():
                # run the model to get the response
                outputs = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=processed_inputs.attention_mask,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    bos_token_id=self.processor.tokenizer.bos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True
                )
            return core.GenerationResp(content=outputs)

    @staticmethod
    def format_conversations(prompt, img_: core._RegularImage):
        conversations = [
            {"role": "<|User|>", "content": f"<image>\n{prompt}", "images": [img_.to_filesys_path()]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        lgr.debug(conversations)
        return conversations

    @staticmethod
    def format_conversations_v1(prompt, img_: core._RegularImage):
        conversations = [
            {"role": "User", "content": f"<image_placeholder>{prompt}", "images": [img_.to_filesys_path()]},
            {"role": "Assistant", "content": ""},
        ]
        lgr.debug(conversations)
        return conversations

    @staticmethod
    def format_conversations_v1_icl(prompt, img_: core._RegularImage, demos: typing.List[core.IclDemo]):
        conversations = []
        for i, demo in enumerate(demos):
            conversations.extend([
                {"role": "User",
                 "content": f"<image_placeholder>{demo.text_in}", "images": [demo.image.to_filesys_path(suffix=i)]},
                {"role": "Assistant",
                 "content": demo.text_out},
            ])

        conversations.extend([
            {"role": "User", "content": f"<image_placeholder>{prompt}", "images": [img_.to_filesys_path()]},
            {"role": "Assistant", "content": ""},
        ])
        lgr.debug(conversations)
        return conversations


class HfProcessorTextExtractor(core.BaseExtractor):

    def __init__(self, processor):
        self.processor = processor

    def extract(self, resp: core.GenerationResp) -> core.TextResult:
        outputs = resp.content[0].cpu().tolist()
        text = self.processor.tokenizer.decode(outputs, skip_special_tokens=True)
        return core.TextResult(text=text)


def main():
    args = util.get_argparser().parse_args()


    image_set = expt.get_image_set(args.image_set, **args.imset_kwargs)
    image_names = image_set.list_image_names()

    question_set = expt.get_question_set(args.question_set)

    lgr.info('Loading model: %s', args.model_name)

    ds2_vlm = DeepSeek2VLM(
        model_name=args.model_name,
        formatter=DsPromptFormatter(),
        image_preprocessor=DsImagePreprocessor())

    text_extractor = HfProcessorTextExtractor(processor=ds2_vlm.processor)

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
                    resp = ds2_vlm.vqa_icl(ques=ques, img=image, demos=demos)
                    record.update(demos=[str(d) for d in demos])
                else:
                    resp = ds2_vlm.vqa(ques=ques, img=image)
                answer = text_extractor.extract(resp)
                lgr.debug(answer)
                record.update(**answer.to_record())

                result_records.append(record)

        util.save_results(args, result_records, ts_str=timestamp)

    lgr.info("Finished %d batches of images with %s", n, args.model_name)


if __name__ == '__main__':
    main()

"""
# moved to Dockerfile
# git clone https://github.com/deepseek-ai/DeepSeek-VL
# pip install -e DeepSeek-VL/

# model_names
# deepseek-vl2-tiny  # gpu cap needed 8.0; got 6.1
# DeepSeek-VL-7B-chat
# https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat
# DeepSeek-VL-1.3B-chat

DEBUG=1 python3 main.py --model_name DeepSeek-VL-7B-chat
"""
