"""InternVL2.5"""

import dataclasses
import logging
import math
import typing
from PIL import Image
import torch
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

import core
import expt
import util

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


lgr = logging


class ItPromptFormatter(core.IPromptFormatter):
    def format(self, ques: core.Question) -> str:
        return ques.text


class ItImagePreprocessor(core.IImagePreprocessor):
    def preprocess(self, image: core._RegularImage):
        return self.load_image(image.to_pil()).to(torch.bfloat16).cuda()

    def load_image(self, image: Image, input_size=448, max_num=12):
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    @staticmethod
    def build_transform(input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform


    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio


class Intern2_5VLM(core.BaseVLM):
    LAYERS_MAP = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}

    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            f'OpenGVLab/{self.model_name}', trust_remote_code=True, use_fast=False)

        self.model = AutoModel.from_pretrained(
            f'OpenGVLab/{self.model_name}',
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
                # If you set `load_in_8bit=True`, you will need one 80GB GPUs.
                # If you set `load_in_8bit=False`, you will need at least two 80GB GPUs.
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=self._split_model(),
            cache_dir=util.HF_CACHE).eval()

        self.generation_config = dict(max_new_tokens=128, do_sample=True,
                                      pad_token_id=self.tokenizer.eos_token_id)

    def _vqa(self, prompt, img_) -> core.BaseResponse:
        lgr.debug(prompt)
        # single-image single-round conversation
        question = f'<image>\n{prompt}'
        response = self.model.chat(self.tokenizer, img_, question, self.generation_config)
        return core.GenerationResp(content=response)

    def _vqa_icl(self, prompt, img, demos):
        raise NotImplementedError("skipped for now")

    def _split_model(self):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = self.LAYERS_MAP[self.model_name]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map



class BasicTextExtractor(core.BaseExtractor):
    def extract(self, resp: core.GenerationResp) -> core.TextResult:
        return core.TextResult(text=resp.content)



def main():
    args = util.get_argparser().parse_args()

    image_set = expt.get_image_set(args.image_set, **args.imset_kwargs)
    image_names = image_set.list_image_names()
    question_set = expt.get_question_set(args.question_set)

    lgr.info('Loading model: %s', args.model_name)

    it_vlm = Intern2_5VLM(
        model_name=args.model_name,
        formatter=ItPromptFormatter(),
        image_preprocessor=ItImagePreprocessor())

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
                resp = it_vlm.vqa(ques=ques, img=image)
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
# InternVL2_5-4B-AWQ  # needs separate package for deployment
# InternVL2_5-4B
# https://huggingface.co/OpenGVLab/InternVL2_5-4B
# InternVL2_5-8B  # not enought GPU memory

DEBUG=1 python3 main.py --model_name InternVL2_5-4B
"""