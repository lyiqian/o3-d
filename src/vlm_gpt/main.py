"""Commercial model: GPT"""

import base64
import functools
import logging
import pathlib
import time
from tqdm import tqdm
import openai

import core
import expt
import util


lgr = logging

class GptFileImagePreprocessor(core.IImagePreprocessor):
    """Upload image if needed and return file ID for GPT API."""

    def __init__(self):
        self.client = openai.OpenAI()

    def preprocess(self, img: core._RegularImage) -> str:
        file_id = self.get_or_create_file(img)
        return file_id

    def get_or_create_file(self, img: core._RegularImage):
        file_id = self._lookup_file_id(img.format_unique_id())
        if file_id is not None:
            return file_id

        return self.create_file(img)

    def create_file(self, img: core._RegularImage) -> str:
        img_uid = img.format_unique_id()

        lgr.info("Uploading file: %s", img_uid)
        file_obj = img.to_bytes()
        assert file_obj.name == img_uid

        uploaded_file = self.client.files.create(
            file=file_obj,
            purpose="vision",
        )
        self._lookup_file_id.cache_clear()
        return uploaded_file.id

    @functools.lru_cache()
    def _lookup_file_id(self, img_uid: str):
        uploaded_files = self.client.files.list().data
        for fobj in uploaded_files:
            if fobj.filename == img_uid:
                return fobj.id
        return None


class GptBase64ImagePreprocessor(core.IImagePreprocessor):
    def preprocess(self, img: core._RegularImage) -> str:
        image_b64 = self.encode(img)
        return image_b64

    @staticmethod
    def encode(img: core._RegularImage) -> str:
        image_file = img.to_bytes()
        image_b64 = base64.b64encode(image_file.read()).decode('utf-8')
        return image_b64


class GptPromptFormatter(core.IPromptFormatter):
    def format(self, ques: core.Question, bbox: bool=False) -> str:
        if bbox:  # unused
            pass
        return ques.text


class GptVLM(core.BaseVLM):
    IMAGE_DETAIL = "high"
    TEMPERATURE = 0.2

    def init_model(self):
        self.client = openai.OpenAI()

    def _vqa(self, prompt, img_: str, bbox=False) -> core.BaseResponse:
        lgr.debug(prompt)
        file_id = img_  # output of GptImagePreprocessor.preprocess

        request = [{
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "file_id": file_id,
                    "detail": self.IMAGE_DETAIL,
                },
                {"type": "input_text", "text": prompt},
            ],
        }]

        err = None
        for attempt in range(3):
            try:
                return self._generate(request)

            except openai.APIError as e:
                err = e
                lgr.error(f"API error on attempt {attempt + 1}: {repr(e)}")
                time.sleep(2 ** attempt)  # exponential backoff
                continue

        else:
            raise err

    def _vqa_icl(self, prompt, img_, demos):
        lgr.debug(prompt)
        messages = []
        for demo in demos:
            img_b64 = GptBase64ImagePreprocessor.encode(demo.image)
            messages.extend([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                        {"type": "text", "text": demo.text_in},
                    ],
                },
                {
                    "role": "assistant",
                    "content": demo.text_out,
                },
            ])

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_}"},
                },
                {"type": "text", "text": prompt},
            ],
        })

        err = None
        for attempt in range(3):
            try:
                return self._generate_chat_completion(messages)

            except openai.APIError as e:
                err = e
                lgr.error(f"API error on attempt {attempt + 1}: {repr(e)}")
                time.sleep(2 ** attempt)  # exponential backoff
                continue

        else:
            raise err

    def _generate(self, request):
        response = self.client.responses.create(
            model=self.model_name,
            input=request,
            temperature=self.TEMPERATURE,
            max_output_tokens=512,
        )
        return core.GenerationResp(content=response)

    def _generate_chat_completion(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.TEMPERATURE,
            max_completion_tokens=512,
        )
        return core.GenerationResp(content=response)



class OpenaiTextExtractor(core.BaseExtractor):

    def extract(self, resp: core.GenerationResp) -> core.TextResult:
        openai_resp = resp.content
        return core.TextResult(text=openai_resp.output_text)

class OpenaiChatExtractor(core.BaseExtractor):

    def extract(self, resp: core.GenerationResp) -> core.TextResult:
        openai_resp = resp.content.choices[0]
        return core.TextResult(text=openai_resp.message.content)


def main():
    args = util.get_argparser().parse_args()

    image_set = expt.get_image_set(args.image_set, **args.imset_kwargs, image_format="bytes")
    image_names = image_set.list_image_names()
    question_set = expt.get_question_set(args.question_set)

    lgr.info('Loading model: %s', args.model_name)

    gpt_vlm = GptVLM(
        model_name=args.model_name,
        formatter=GptPromptFormatter(),
        image_preprocessor=_get_img_preproc(args),
    )
    text_extractor = _get_text_extractor(args)

    timestamp, result_records = util.prepare_vlm_run(args)

    batches_of_names = util.batches_of(image_names, size=100)
    for n, names in enumerate(batches_of_names, start=1):
        lgr.info("Processing batch %d", n)
        for image_fname in tqdm(names, desc=f"Batch {n}", total=len(names)):
            kwargs = expt.prep_question_kwargs(question_set, image_set, image_fname)
            questions = question_set.list_questions(**kwargs)  # might have rand MCQs

            if _to_skip_image(args, image_fname, len(questions), result_records):
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
                    resp = gpt_vlm.vqa_icl(ques=ques, img=image, demos=demos)
                    record.update(demos=[str(d) for d in demos])
                else:
                    resp = gpt_vlm.vqa(ques=ques, img=image)
                answer = text_extractor.extract(resp)
                lgr.debug(answer)
                record.update(**answer.to_record())

                result_records.append(record)

        util.save_results(args, result_records, ts_str=timestamp)

    lgr.info("Finished %d batches of images with %s", n, args.model_name)

def _get_img_preproc(args):
    if args.icl:
        return GptBase64ImagePreprocessor()
    else:
        return GptFileImagePreprocessor()


def _get_text_extractor(args):
    if args.icl:
        return OpenaiChatExtractor()
    else:
        return OpenaiTextExtractor()

def _to_skip_image(args, image_fname, n_ques, result_records):
    if not args.resume:
        return False

    matched_records = []
    for r in result_records:
        if r["image_name"] == image_fname:
            matched_records.append(r)

    if len(matched_records) == n_ques:  # all questions answered
        return True

    if len(matched_records) > 0:  # partially completed
        for old_record in matched_records:
            result_records.remove(old_record)

    return False


if __name__ == '__main__':
    main()


"""
# model_names
# gpt-4.1-mini

DEBUG=1 python3 main.py --model_name gpt-4.1-mini
"""