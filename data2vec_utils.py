import torch
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput
from transformers import ViTConfig
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, Trainer

import logger

from dataclasses import dataclass

from typing import Optional

@dataclass
class Data2VecEncoderOutput(BaseModelOutput):
    data2vec_target: Optional[torch.Tensor] = None


class Data2VecOutput(ModelOutput):
    """
    Class for data2vec model outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Data2vec loss.
        target (`torch.FloatTensor` of shape `(batch_size, n_patches, embed_dim)`):
            Averaged teacher hidden layer representations.
        prediction (`torch.FloatTensor` of shape `(batch_size, n_patches, embed_dim)`):
            Student prediction of target.
    """
    loss: torch.FloatTensor
    target: Optional[torch.FloatTensor]
    prediction: Optional[torch.FloatTensor]


class ViTConfigForData2Vec(ViTConfig):
    def __init__(self, n_layers_to_average=None, huber_loss_delta=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_layers_to_average = n_layers_to_average
        self.huber_loss_delta = huber_loss_delta


class TeacherUpdateCallback(TrainerCallback):
    """Callback which performs ema-update of the teacher with constant momentum coef"""

    def __init__(self, model: torch.nn.Module, momentum: float):
        self.model = model
        self.momentum = momentum
        
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.model.update_teacher(self.momentum)


class NirvanaCheckpointTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)
        if self.is_world_process_zero():
            try:
                import nirvana_dl.snapshot as snap
                snap.dump_snapshot()
                logger.info('Checkpoint saved to snapshots.')
            except Exception as e:
                logger.info(f'Checkpoint not saved to snapshots: {e}')