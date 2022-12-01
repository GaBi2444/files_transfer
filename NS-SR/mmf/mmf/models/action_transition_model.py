# Copyright (c) Facebook, Inc. and its affiliates.

# Initial version was taken from https://github.com/uclanlp/visualbert
# which was cleaned up and adapted for MMF.

import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.modules.embeddings import SituationGraphEmbeddings
from mmf.modules.hf_layers import BertEncoderJit, BertLayerJit
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.modeling import get_optimizer_parameters_for_bert
from mmf.utils.transform import (
    transform_to_batch_sequence,
    transform_to_batch_sequence_dim,
)
from omegaconf import OmegaConf
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertConfig,
    BertForPreTraining,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
)

import torchvision.models as models

class ATM_Base(BertPreTrainedModel):
    def __init__(
        self,
        config,
        embedding_strategy="plain",
        bypass_transformer=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        super().__init__(config)
        self.config = config
        config.embedding_strategy = embedding_strategy
        config.bypass_transformer = bypass_transformer
        config.output_attentions = output_attentions
        config.output_hidden_states = output_hidden_states

        #self.embeddings = BertVisioLinguisticEmbeddings(config) 
        self.embeddings = SituationGraphEmbeddings(config)
        # goal: input--> embedding viusal / language --> reorganize
        self.encoder = BertEncoderJit(config)
        self.pooler = BertPooler(config)
        # 
        self.bypass_transformer = config.bypass_transformer

        if self.bypass_transformer:
            self.additional_layer = BertLayerJit(config)

        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.init_weights()

    def forward(
        self,
        act_id, obj1_id, rel_id, obj2_id, special_id,
        frame_feature, obj1_feature, obj2_feature,
        special_semantic, act_semantic, obj1_senmatic, rel_semantic, obj2_senmatic,
        attention_mask,
        type_token, hyper_token, triplet_token, situation_token
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(attention_mask)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # Python builtin next is currently not supported in Torchscript
        if not torch.jit.is_scripting():
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
        act_id, obj1_id, rel_id, obj2_id, special_id,
        frame_feature, obj1_feature, obj2_feature,
        special_semantic, act_semantic, obj1_senmatic, rel_semantic, obj2_senmatic,
        type_token, hyper_token, triplet_token, situation_token
        )
        #print('embedding_output',embedding_output)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask)
        #print('encoded_layers',encoded_layers)
        sequence_output = encoded_layers[0]
        pooled_output = self.pooler(sequence_output)
        attn_data_list = []

        if not torch.jit.is_scripting():
            if self.output_attentions:
                attn_data_list = encoded_layers[1:]
        else:
            assert (
                not self.output_attentions
            ), "output_attentions not supported in script mode"

        return sequence_output, pooled_output, attn_data_list


class AMTForPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        # If bert_model_name is not specified, you will need to specify
        # all of the required parameters for BERTConfig and a pretrained
        # model won't be loaded
        self.bert_model_name = getattr(self.config, "bert_model_name", None)
        self.bert_config = BertConfig.from_dict(
            OmegaConf.to_container(self.config, resolve=True)
        )

        self.bert = ATM_Base(
                self.bert_config,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )

        self.vocab_size = self.config.vocab_size

        bert_masked_lm = BertForPreTraining(self.bert.config)

        self.cls = deepcopy(bert_masked_lm.cls)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()
                self.cls.apply(self.bert._init_weights)

            self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them
            instead.
        """
        self.bert._tie_or_clone_weights(
            self.cls.predictions.decoder, self.bert.embeddings.word_embeddings
        )

    def forward(
        self,
        act_id,act_target,act_attmask,
        obj1_id,obj1_target,obj1_attmask,
        rel_id,rel_target,rel_attmask,
        obj2_id,obj2_target,obj2_attmask,
        special_id,special_target,special_attmask,
        frame_feature, frame_attemask,
        obj1_feature,obj2_feature,
        special_semantic, act_semantic, obj1_senmatic, rel_semantic, obj2_senmatic,
        type_token, hyper_token, triplet_token, situation_token
    ) -> Dict[str, Tensor]:

        b, max_frame, max_act = act_attmask.shape
        _, _, max_rel = rel_attmask.shape

        #mask = torch.stack([special_attmask,obj1_attmask,rel_attmask,obj2_attmask], dim=-1).reshape(b, max_frame, 4 * max_rel)
        mask = torch.stack([obj1_attmask,rel_attmask,obj2_attmask], dim=-1).reshape(b, max_frame, 3 * max_rel)
        #print('mask',mask.shape)
        #print('act_attmask', act_attmask.shape)
        #print('frame_attemask',frame_attemask.shape)
        #attention_mask = torch.cat([act_attmask, mask, frame_attemask.unsqueeze(-1)], dim=-1).reshape(b,-1)
        attention_mask = torch.cat([act_attmask, mask], dim=-1).reshape(b,-1)

        #print('attention_mask',attention_mask.shape)
        #print(attention_mask[0])

        sequence_output, pooled_output, attention_weights = self.bert(
            act_id, obj1_id, rel_id, obj2_id, special_id,
            frame_feature, obj1_feature,obj2_feature,
            special_semantic, act_semantic, obj1_senmatic, rel_semantic, obj2_senmatic,
            attention_mask,
            type_token, hyper_token, triplet_token, situation_token
        )

        output_dict: Dict[str, Tensor] = {}
        if not torch.jit.is_scripting():
            if self.output_attentions:
                output_dict["attention_weights"] = attention_weights

            if self.output_hidden_states:
                output_dict["sequence_output"] = sequence_output
                output_dict["pooled_output"] = pooled_output
        else:
            assert not (
                self.output_attentions or self.output_hidden_states
            ), "output_attentions or output_hidden_states not supported in script mode"

        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        #print('prediction_scores',prediction_scores.shape)

        scores = torch.chunk(prediction_scores, max_frame ,dim=1)

        # act_score = torch.stack([s[:,:max_act,:] for s in scores],dim=1)
        # obj1_score = torch.stack([s[:,max_act+1:-1:4,:] for s in scores],dim=1)
        # rel_score = torch.stack([s[:,max_act+2:-1:4,:] for s in scores],dim=1)
        # obj2_score = torch.stack([s[:,max_act+3:-1:4,:] for s in scores],dim=1)

        act_score = torch.stack([s[:,:max_act,:] for s in scores],dim=1)
        obj1_score = torch.stack([s[:,max_act::3,:] for s in scores],dim=1)
        rel_score = torch.stack([s[:,max_act+1::3,:] for s in scores],dim=1)
        obj2_score = torch.stack([s[:,max_act+2::3,:] for s in scores],dim=1)

        #print('prediction_scores',prediction_scores.shape)
        
        output_dict["logits"] = prediction_scores

        masked_act_loss = self.loss_fct(
                act_score.contiguous().view(-1, self.vocab_size),
                act_target.contiguous().view(-1),
            )
        masked_obj1_loss = self.loss_fct(
                obj1_score.contiguous().view(-1, self.vocab_size),
                obj1_target.contiguous().view(-1),
            )
        masked_rel_loss = self.loss_fct(
                rel_score.contiguous().view(-1, self.vocab_size),
                rel_target.contiguous().view(-1),
            )
        
        masked_obj2_loss = self.loss_fct(
                obj2_score.contiguous().view(-1, self.vocab_size),
                obj2_target.contiguous().view(-1),
            )
            #print('masked_lm_loss',masked_lm_loss)
        output_dict["masked_act_loss"] = masked_act_loss
        output_dict["masked_obj1_loss"] = masked_obj1_loss
        output_dict["masked_rel_loss"] = masked_rel_loss
        output_dict["masked_obj2_loss"] = masked_obj2_loss

        output_dict["loss"] = 0.6*masked_act_loss + 0.1*masked_obj1_loss + 0.2*masked_rel_loss + 0.1*masked_obj2_loss

        #_, index = torch.max(prediction_scores,-1)

        _, act_index = torch.max(act_score.contiguous().view(-1, self.vocab_size),-1)
        _, obj1_index = torch.max(obj1_score.contiguous().view(-1, self.vocab_size),-1)
        _, rel_index = torch.max(rel_score.contiguous().view(-1, self.vocab_size),-1)
        _, obj2_index = torch.max(obj2_score.contiguous().view(-1, self.vocab_size),-1)
        act_index = act_index.reshape(b,max_frame, max_act)
        obj1_index = obj1_index.reshape(b,max_frame, max_rel)
        rel_index = rel_index.reshape(b,max_frame, max_rel)
        obj2_index = obj2_index.reshape(b,max_frame, max_rel)

        output_dict['pre_act_cls'] = act_index
        output_dict['pre_obj1_cls'] = obj1_index
        output_dict['pre_rel_cls'] = rel_index
        output_dict['pre_obj2_cls'] = obj2_index

        return output_dict

@registry.register_model("action_transition_model")
class SRTransformer(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @classmethod
    def config_path(cls):
        return "mmf/configs/models/atm/pretrain.yaml"

    def build(self):

        self.model = AMTForPretraining(self.config)

        if getattr(self.config, "freeze_base", False):
            for p in self.model.bert.parameters():
                p.requires_grad = False


    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)

    # Backward compatibility for code from original VisualBERT
    @classmethod
    def format_state_key(cls, key):
        return (
            key.replace("bert.bert", "model.bert")
            .replace("bert.cls", "model.cls")
            .replace("bert.classifier", "model.classifier")
        )

    def forward(self, sample_list):

        output_dict = self.model(
            sample_list.act_id,sample_list.act_target,sample_list.act_attmask,
            sample_list.obj1_id,sample_list.obj1_target,sample_list.obj1_attmask,
            sample_list.rel_id,sample_list.rel_target,sample_list.rel_attmask,
            sample_list.obj2_id,sample_list.obj2_target,sample_list.obj2_attmask,
            sample_list.special_id,sample_list.special_target,sample_list.special_attmask,
            sample_list.frame_feature, sample_list.frame_attmask,
            sample_list.obj1_feature, sample_list.obj2_feature,
            sample_list.special_semantic, sample_list.act_semantic, sample_list.obj1_senmatic, sample_list.rel_semantic, sample_list.obj2_senmatic, 
            sample_list.type_token, sample_list.hyper_token, sample_list.triplet_token, sample_list.situation_token
        )

        output_dict['question_id'] = sample_list.question_id
        output_dict['select_keyframe'] = sample_list.select_keyframe

        if "pretraining" in self.config.training_head_type:
            loss_key = "{}/{}".format(
                sample_list.dataset_name, sample_list.dataset_type
            )
            output_dict["losses"] = {}
            output_dict["losses"][loss_key + "/masked_act_loss"] = output_dict.pop(
                "masked_act_loss"
            )

            output_dict["losses"][loss_key + "/masked_obj1_loss"] = output_dict.pop(
                "masked_obj1_loss"
            )

            output_dict["losses"][loss_key + "/masked_rel_loss"] = output_dict.pop(
                "masked_rel_loss"
            )

            output_dict["losses"][loss_key + "/masked_obj2_loss"] = output_dict.pop(
                "masked_obj2_loss"
            )

        return output_dict
