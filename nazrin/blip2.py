import csv
import glob
import json
import os
from pathlib import Path

import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from tqdm import tqdm

from nazrin.config.blip2 import Blip2Config

MODELS = [
    {"name": "blip2_t5", "type": "pretrain_flant5xxl"},
    {"name": "blip2_opt", "type": "pretrain_opt2.7b"},
    {"name": "blip2_opt", "type": "pretrain_opt6.7b"},
    {"name": "blip2_opt", "type": "caption_coco_opt2.7b"},
    {"name": "blip2_opt", "type": "caption_coco_opt6.7b"},
    {"name": "blip2_t5", "type": "pretrain_flant5xl"},
    {"name": "blip2_t5", "type": "caption_coco_flant5xl"},
]


model_namelist, model_typelist, model_history = [], [], ""
model, vis_processors = "", {}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_model_type() -> list:
    for model in MODELS:
        model_typelist.append(model["type"])
    return model_typelist


def load_model(model_type):
    global model_history
    name, modeltype = (
        [model["name"] for model in MODELS if model["type"] == model_type][0],
        model_type,
    )

    if modeltype != model_history:
        global model, vis_processors

        print(f"loading model {modeltype}...")

        model, vis_processors, _ = load_model_and_preprocess(
            name=name, model_type=modeltype, is_eval=True, device=get_device()
        )

        model_history = modeltype

    else:
        pass

    return model, vis_processors


def gen_caption(image, caption_type, length_penalty, repetition_penalty, temperature):
    device = get_device()
    try:
        image = vis_processors["eval"](image).unsqueeze(0).to(device)
    except KeyError:
        print("Please select models!")
    else:
        if caption_type == "Beam Search":
            caption = model.generate(
                {"image": image},
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
            )
        else:
            caption = model.generate(
                {"image": image},
                use_nucleus_sampling=True,
                num_captions=3,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
            )

        caption = "\n".join(caption)

        return caption


def blip2(config_file):
    config = Blip2Config.parse_auto(config_file)

    load_model(config.model_type)

    if config.recursive:
        glob_str = f"{config.directory}/**/*"
    else:
        glob_str = f"{config.directory}/*"

    for file_path in tqdm(glob.glob(glob_str, recursive=config.recursive)):
        meta_path = os.path.splitext(file_path)[0] + ".json"
        try:
            image = Image.open(file_path).convert("RGB")
        except:
            continue

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.loads(f.read())
        else:
            meta = {}

        caption = gen_caption(
            image,
            config.caption_type,
            config.length_penalty,
            config.repetition_penalty,
            config.temperature,
        )

        meta["caption"] = caption

        with open(meta_path, "w") as f:
            f.write(json.dumps(meta, indent=4))
