#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import logging
import os
from statistics import mode
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

import transformers
from transformers import (
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from data2vec_utils import TeacherUpdateCallback
from data2vec_model import ViTForData2Vec, ViTConfigForData2Vec
from data_processing.yt_dataset import Data2VecImagenetIterableYTDataset

""" Pre-training a 🤗 Transformers model for simple masked image modeling (SimMIM).
Any model supported by the AutoModelForMaskedImageModeling API can be used.
"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.18.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    dataset_name: Optional[str] = field(
        default="cifar10", metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    image_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "The column name of the images in the files. If not set, will try to use 'image' or 'img'."},
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    mask_patch_size: int = field(default=32, metadata={"help": "The size of the square patches to use for masking."})
    mask_ratio: float = field(
        default=0.6,
        metadata={"help": "Percentage of patches to mask."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    num_readers: int = field(
        default=1,
        metadata={"help": "Number of reading processes for YT dataset"}
    )

    def __post_init__(self):
        data_files = dict()
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/feature extractor we are going to pre-train.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Can be a local path to a pytorch_model.bin or a "
            "checkpoint identifier on the hub. "
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store (cache) the pretrained models/datasets downloaded from the hub"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    image_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "The size (resolution) of each image. If not specified, will use `image_size` of the configuration."
        },
    )
    patch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "The size (resolution) of each patch. If not specified, will use `patch_size` of the configuration."
        },
    )
    encoder_stride: Optional[int] = field(
        default=None,
        metadata={"help": "Stride to use for the encoder."},
    )
    n_layers_to_average: Optional[int] = field(
        default=None,
        metadata={"help": "Amount of layers averaged for creating a target."}
    )
    huber_loss_delta: float = field(
        default=2.0,
        metadata={"help": "Delta coefficient in Huber loss, computed between student and teacher predictions."}
    )
    momentum: float = field(
        default=0.9998,
        metadata={"help": "Momentum coefficient for teacher updates."}
    )
    experiment_name: str = field(
        default=None,
        metadata={"help": "Unique name for a checkpoint."}
    )


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    mask = torch.stack([example["mask"] for example in examples])
    return {"pixel_values": pixel_values, "bool_masked_pos": mask}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        print("Model:\n", model_args)
        print("Data:\n", data_args)
        print("Training:\n", training_args)


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Create config
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = ViTConfigForData2Vec(
        n_layers_to_average=model_args.n_layers_to_average,
        huber_loss_delta=model_args.huber_loss_delta
    )
    config.update(config_kwargs)

    # adapt config
    model_args.image_size = model_args.image_size if model_args.image_size is not None else config.image_size
    model_args.patch_size = model_args.patch_size if model_args.patch_size is not None else config.patch_size
    model_args.encoder_stride = (
        model_args.encoder_stride if model_args.encoder_stride is not None else config.encoder_stride
    )

    config.update(
        {
            "image_size": model_args.image_size,
            "patch_size": model_args.patch_size,
            "encoder_stride": model_args.encoder_stride,
        }
    )

    logger.info(f"Using config: {config}")

    # create model
    model = ViTForData2Vec(config)

    # Initialize our dataset.
    train_dataset = Data2VecImagenetIterableYTDataset(root=data_args.train_dir, 
        image_size=model_args.image_size, 
        model_patch_size=model_args.patch_size,
        mask_patch_size=data_args.mask_patch_size,
        mask_ratio=data_args.mask_ratio,
        num_readers=data_args.num_readers,
    ) 
    val_dataset = Data2VecImagenetIterableYTDataset(root=data_args.validation_dir,
        image_size=model_args.image_size, 
        model_patch_size=model_args.patch_size,
        mask_patch_size=data_args.mask_patch_size,
        mask_ratio=data_args.mask_ratio,
        num_readers=data_args.num_readers,
    )

    ds = {
        "train": train_dataset,
        "validation": val_dataset,
    }

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in ds.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        raise RuntimeError("Train-val split is not supported")
        split = ds["train"].train_test_split(data_args.train_val_split)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]


    if training_args.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            ds["train"] = ds["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        # Transforms already set in the dataset
        # ds["train"].set_transform(preprocess_images)

    if training_args.do_eval:
        if "validation" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            ds["validation"] = (
                ds["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Transforms already set in the dataset
        # ds["validation"].set_transform(preprocess_images)

    # Initialize our trainer

    opt = torch.optim.Adam(model.student.parameters(), 
        lr=training_args.learning_rate, 
        weight_decay=training_args.weight_decay
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(opt, 
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        optimizers=(opt, scheduler),
        data_collator=collate_fn,
        callbacks=[TeacherUpdateCallback(model, model_args.momentum)],
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "masked-image-modeling",
        "dataset": data_args.dataset_name,
        "tags": ["masked-image-modeling"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    if model_args.experiment_name:
        exp_name = model_args.experiment_name
        torch.save(model.state_dict(), f"checkpoints/{exp_name}_model.pth")
        torch.save(opt.state_dict(), f"checkpoints/{exp_name}_optimizer.pth")

if __name__ == "__main__":
    main()