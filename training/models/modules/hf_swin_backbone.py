from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import SwinPreTrainedModel, SwinConfig
from transformers.modeling_outputs import BackboneOutput
from transformers.models.swin.modeling_swin import SwinEmbeddings, SwinEncoder
# from transformers.utils import BackboneMixin


@dataclass
class SwinBackboneOutput(BackboneOutput):
    unnormed_layer3: torch.Tensor = None


class HFSwinBackbone(SwinPreTrainedModel):
    """Modified version of backbone that supports masked tokens"""

    def __init__(self, config: SwinConfig):
        super().__init__(config)
        super()._init_backbone(config)

        self.lr_tunable = True

        self.num_features = [config.embed_dim] + [
            int(config.embed_dim * 2**i) for i in range(len(config.depths))
        ]
        self.embeddings = SwinEmbeddings(config, use_mask_token=True)
        self.encoder = SwinEncoder(config, self.embeddings.patch_grid)

        # Add layer norms to hidden states of out_features
        hidden_states_norms = {}
        for stage, num_channels in zip(self._out_features, self.channels):
            hidden_states_norms[stage] = nn.LayerNorm(num_channels)
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    @staticmethod
    def from_config(config: SwinConfig):
        return HFSwinBackbone._from_config(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        Modified version that also includes masking

        Returns:
            features
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        embedding_output, input_dimensions = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos
        )

        outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            always_partition=True,
            return_dict=True,
        )

        hidden_states = outputs.reshaped_hidden_states

        feature_maps = ()
        unnormed_stage3 = None
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                batch_size, num_channels, height, width = hidden_state.shape
                hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
                hidden_state = hidden_state.view(
                    batch_size, height * width, num_channels
                )
                if stage == "stage3":
                    unnormed_stage3 = hidden_state
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                hidden_state = hidden_state.view(
                    batch_size, height, width, num_channels
                )
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps += (hidden_state,)

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        return SwinBackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
            unnormed_layer3=unnormed_stage3,
        )

    def get_parameters(self, base_lr_dict: dict):
        has_decay = []
        has_decay_names = []
        no_decay = []
        no_decay_names = []

        for name, param in self.named_parameters():
            if (
                len(param.shape) == 1
                or name.endswith(".bias")
                or check_keywords_in_name(name, {"relative_position_bias_table"})
            ):
                no_decay.append(param)
                no_decay_names.append(name)
            else:
                has_decay.append(param)
                has_decay_names.append(name)

        if "lr" in base_lr_dict:
            lr = {"lr": base_lr_dict["lr"]}
        else:
            lr = {}
        # print(f"No decay: {no_decay_names}")

        # base_lr dict in first case with optional lr and weight decay, lr dict only with LR
        return [
            {"params": has_decay, **base_lr_dict},
            {"params": no_decay, "weight_decay": 0.0, **lr},
        ]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
