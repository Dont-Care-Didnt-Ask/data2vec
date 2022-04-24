import torch
import torch.distributed as dist
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput
from transformers import ViTConfig
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, Trainer

import logging

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
        teacher_hidden_states: (`tuple` of `torch.FloatTensor` of shape `(batch_size, n_patches, embed_dim)` of length n_layers + 1):
            Teacher hidden states.
        student_hidden_states: (`tuple` of `torch.FloatTensor` of shape `(batch_size, n_patches, embed_dim)` of length n_layers + 1):
            Student hidden states.
        teacher_attentions: (`tuple` of `torch.FloatTensor` of shape `(batch_size, num_heads, n_patches, n_patches)` of length n_layers):
            Teacher attentions.
        student_attentions: (`tuple` of `torch.FloatTensor` of shape `(batch_size, num_heads, n_patches, n_patches)` of length n_layers):
            Student attentions.
    """
    loss: torch.FloatTensor
    target: Optional[torch.FloatTensor] = None
    prediction: Optional[torch.FloatTensor] = None
    teacher_hidden_states: Optional[torch.Tensor] = None
    student_hidden_states: Optional[torch.Tensor] = None
    teacher_attentions: Optional[torch.Tensor] = None
    student_attentions: Optional[torch.Tensor] = None


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


class CheckpointTeacher(TrainerCallback):
    def __init__(self, model: torch.nn.Module, save_each_n_epochs: int):
        self.model = model
        self.save_each_n_epochs = save_each_n_epochs
        self.current_epoch = 0
    
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        self.current_epoch += 1

        if (self.current_epoch + 1) % self.save_each_n_epochs == 0 and dist.get_rank() == 0:
            path = args.output_dir + f"/teacher_epoch_{self.current_epoch}.pth"
            torch.save({"teacher_encoder": self.model.teacher.state_dict(), 
                        "embeddings": self.model.embeddings.state_dict()}, path)


logger = logging.getLogger(__name__)

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