import glob
import json
import os
from typing import *

import cv2
import huggingface_hub
import numpy as np
import pandas as pd
import pillow_avif
import torch
import torch.utils.data
import tqdm
from PIL import Image
from tensorflow import keras

from nazrin.config.wd14tag import WD14TagConfig
from nazrin.types import ImageMeta

HF_TOKEN = os.environ.get("HF_TOKEN", "")

FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
IMAGE_SIZE = 448

MODEL_DIR = "./models/wd14tagger"


def load_model(model_repo: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    for file in FILES:
        huggingface_hub.hf_hub_download(
            model_repo,
            file,
            use_auth_token=HF_TOKEN,
            cache_dir=MODEL_DIR,
            force_filename=file,
        )
    for file in SUB_DIR_FILES:
        huggingface_hub.hf_hub_download(
            model_repo,
            file,
            subfolder=SUB_DIR,
            use_auth_token=HF_TOKEN,
            cache_dir=os.path.join(MODEL_DIR, SUB_DIR),
            force_filename=file,
        )
    model = keras.models.load_model(MODEL_DIR)
    labels = pd.read_csv(os.path.join(MODEL_DIR, FILES[-1]))["name"].tolist()
    return model, labels


def preprocess_image(
    image: Image.Image,
):
    image = image.convert("RGB")
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(
        image,
        ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)),
        mode="constant",
        constant_values=255,
    )

    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

    image = image.astype(np.float32)
    return image


class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(self, meta_paths: List[str]):
        self.meta_paths = meta_paths

    def __len__(self):
        return len(self.meta_paths)

    def __getitem__(self, idx):
        meta_path = str(self.meta_paths[idx])

        with open(meta_path, "r") as f:
            meta: ImageMeta = json.loads(f.read())

        image = Image.open(os.path.join(os.path.dirname(meta_path), meta["filename"]))
        image = preprocess_image(image)

        return image, meta, meta_path


def collate_fn_remove_corrupted(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def tag(config_file: str):
    config = WD14TagConfig.parse_auto(config_file)
    if config.recursive:
        glob_str = f"{config.directory}/**/*.json"
    else:
        glob_str = f"{config.directory}/*.json"

    model, labels = load_model(config.tagger_repo)
    dataset = ImageLoadingPrepDataset(glob.glob(glob_str, recursive=config.recursive))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.max_workers,
        collate_fn=collate_fn_remove_corrupted,
        drop_last=False,
    )

    for data_entry in tqdm.tqdm(dataloader):
        images = np.array([x[0] for x in data_entry])
        probs = model(images, training=False)
        probs = probs.numpy()

        for (_, meta, meta_path), prob in zip(data_entry, probs):
            l = list(zip(labels, prob.astype(float)))

            ratings_names = l[:4]

            tags_names = l[4:]
            res = [x for x in tags_names if x[1] > config.score_threshold]
            res = dict(res)

            b = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))

            meta["wd14_tags"] = list(b.keys())

            with open(meta_path, "w") as f:
                f.write(json.dumps(meta))
