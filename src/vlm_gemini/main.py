"""Commercial model: Gemini"""

import logging
import time
from tqdm import tqdm

from google import genai
from google.genai import types

import core
import expt
import util


lgr = logging

class GmnImagePreprocessor(core.IImagePreprocessor):
    def preprocess(self, img: core.FileSysImage):
        return img.to_pil()


class GmnPromptFormatter(core.IPromptFormatter):
    def format(self, ques: core.Question) -> str:
        return ques.text


class GmnVLM(core.BaseVLM):
    TEMPERATURE = 0.1
    RESP_MIME_TYPE = "text/plain"

    def init_model(self):
        self.client = genai.Client()  # uses GEMINI_API_KEY env var

    def _vqa(self, prompt, img_: str) -> core.BaseResponse:
        lgr.debug(prompt)

        contents = [img_, prompt]

        err = None
        for attempt in range(10):
            try:
                return self._generate(contents)

            except genai.errors.APIError as e:
                err = e
                lgr.error(f"API error on attempt {attempt + 1}: {repr(e)}")
                time.sleep(2 ** attempt)  # exponential backoff
                continue

        else:
            raise err

    def _vqa_icl(self, prompt, img_, demos):
        lgr.debug(prompt)
        contents = []
        for demo in demos:
            contents.append(demo.image.to_pil())
            contents.append(demo.text_in)
            contents.append(demo.text_out)

        contents.append(img_)
        contents.append(prompt)

        err = None
        for attempt in range(3):
            try:
                return self._generate(contents)

            except genai.errors.APIError as e:
                err = e
                lgr.error(f"API error on attempt {attempt + 1}: {repr(e)}")
                time.sleep(2 ** attempt)  # exponential backoff
                continue

        else:
            raise err

    def _generate(self, contents):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=self.TEMPERATURE,
                response_mime_type=self.RESP_MIME_TYPE,
            )
        )
        return core.GenerationResp(content=response)


class GeminiTextPlainExtractor(core.BaseExtractor):

    def extract(self, resp: core.GenerationResp) -> core.TextResult:
        return core.TextResult(text=resp.content.text.strip())


def main():
    args = util.get_argparser().parse_args()

    image_set = expt.get_image_set(args.image_set, **args.imset_kwargs)
    image_names = image_set.list_image_names()
    question_set = expt.get_question_set(args.question_set)

    lgr.info('Loading model: %s', args.model_name)

    gmn_vlm = GmnVLM(
        model_name=args.model_name,
        formatter=GmnPromptFormatter(),
        image_preprocessor=GmnImagePreprocessor())

    text_extractor = GeminiTextPlainExtractor()

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
                    resp = gmn_vlm.vqa_icl(ques=ques, img=image, demos=demos)
                    record.update(demos=[str(d) for d in demos])
                else:
                    resp = gmn_vlm.vqa(ques=ques, img=image)
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
# gemini-2.5-flash
# gemini-2.5-flash-lite
# gemini-3.5-flash

DEBUG=1 python3 main.py --model_name gemini-2.5-flash-lite
"""
