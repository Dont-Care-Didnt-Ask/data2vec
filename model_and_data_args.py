from curses import meta
from dataclasses import dataclass, field
from typing import Optional

# Disclaimer: most of this code was stolen from 
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-pretraining/run_mim.py,
# and some fields of the following classes are either not used or used for other purposes.
# Therefore, "help" may be misleading. 

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    dataset_class: Optional[str] = field(
        default=None, 
        metadata={"help": "Class to use for a dataset creation (e.g. ImageFolder, Data2VecImagenetIterableYTDataset, etc.)"}
    )
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
    masking_strategy: str = field(
        default="beit",
        metadata={"help": "The masking strategy (beit-like, random, etc.)"}
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
        #metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
        metadata={"help": "If training from scratch, pass a model type from the list: <the list is empty yet>"},
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
