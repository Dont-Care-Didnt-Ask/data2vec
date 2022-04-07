from curses import use_default_colors
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer, ViTPreTrainedModel

import torch
import torch.nn as nn
import torch.nn.functional as F

from data2vec_utils import Data2VecEncoderOutput, Data2VecOutput, ViTConfigForData2Vec

class ViTEncoderForData2Vec(nn.Module):
    supports_data2vec_target = True    

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.supports_data2vec_target = True

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        compute_data2vec_target=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        data2vec_target = torch.empty_like(hidden_states) if compute_data2vec_target else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if compute_data2vec_target and (i >= self.config.num_hidden_layers - self.config.n_layers_to_average):
                normed_state = F.layer_norm(hidden_states, hidden_states.shape[-1:])
                averaging_step = i - (self.config.num_hidden_layers - self.config.n_layers_to_average)
                
                data2vec_target.mul_(averaging_step / (averaging_step + 1)).add_(normed_state, alpha=1. / (averaging_step + 1))

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        return Data2VecEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            data2vec_target=data2vec_target
        )


class ViTForData2Vec(ViTPreTrainedModel):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config

        # TODO: init same
        self.embeddings = ViTEmbeddings(config, use_mask_token=True)
        self.student = ViTEncoderForData2Vec(config)
        self.teacher = ViTEncoderForData2Vec(config)

        self.post_init()

        # copy student parameters to teacher
        self.update_teacher(0.)

        for p in self.teacher.parameters():
            p.requires_grad = False

    def update_teacher(self, momentum):
        for param_student, param_teacher in zip(self.student.parameters(), self.teacher.parameters()):
            if param_student.requires_grad:
                param_teacher.data.mul_(momentum).add_((1 - momentum) * param_student.detach().data)

    def forward(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=True,
    ):
        assert return_dict, "return_dict=False is not supported yet"
        assert head_mask is None, "head_mask is not supported yet"
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        ## head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        with torch.no_grad():
            embedding_output = self.embeddings(
                pixel_values,
                bool_masked_pos=None,
                interpolate_pos_encoding=interpolate_pos_encoding
            )

            teacher_outputs = self.teacher(
                embedding_output,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                compute_data2vec_target=True
            )
            target = teacher_outputs.data2vec_target

        masked_embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding
        )

        student_outputs = self.student(
            masked_embedding_output, 
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            compute_data2vec_target=False
        )

        prediction = student_outputs.last_hidden_state

        loss = F.huber_loss(prediction, target, delta=self.config.huber_loss_delta)

        return Data2VecOutput(
            loss=loss,
            target=target,
            prediction=prediction,
            teacher_hidden_states=teacher_outputs.hidden_states,
            teacher_attentions=teacher_outputs.attentions,
            student_hidden_states=student_outputs.hidden_states,
            student_attentions=student_outputs.attentions
        )