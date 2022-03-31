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

import scipy.stats
import logging
import os
import sys

import numpy as np
import torch

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from data2vec_utils import TeacherUpdateCallback, NirvanaCheckpointTrainer
from data2vec_model import ViTForData2Vec, ViTConfigForData2Vec
from data_processing.data2vec_datasets import build_data2vec_dataset, data2vec_collator
from model_and_data_args import ModelArguments, DataTrainingArguments

""" Pre-training a 🤗 Transformers model for simple masked image modeling (SimMIM).
Any model supported by the AutoModelForMaskedImageModeling API can be used.
"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.18.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")

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
    train_path = data_args.train_dir
    val_path = data_args.validation_dir

    ds = {
        "train": build_data2vec_dataset(train_path, data_args, model_args) if training_args.do_train else None,
        "validation": build_data2vec_dataset(val_path, data_args, model_args) if training_args.do_eval else None,
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

    trainer = NirvanaCheckpointTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        optimizers=(opt, scheduler),
        data_collator=data2vec_collator,
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


if __name__ == "__main__":
    main()