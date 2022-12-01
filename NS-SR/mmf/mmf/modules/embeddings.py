# Copyright (c) Facebook, Inc. and its affiliates.
# TODO: Update kwargs with defaults

import os
import pickle
from copy import deepcopy
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import torch
from mmf.modules.attention import AttentionLayer, SelfAttention, SelfGuidedAttention
from mmf.modules.bottleneck import MovieBottleneck
from mmf.modules.layers import AttnPool1d, Identity, PoseConvNet
from mmf.utils.file_io import PathManager
from mmf.utils.vocab import Vocab
from torch import Tensor, nn
from transformers.modeling_bert import BertEmbeddings


class TextEmbedding(nn.Module):
    def __init__(self, emb_type, **kwargs):
        super().__init__()
        self.model_data_dir = kwargs.get("model_data_dir", None)
        self.embedding_dim = kwargs.get("embedding_dim", None)

        # Update kwargs here
        if emb_type == "identity":
            self.module = Identity()
            self.module.text_out_dim = self.embedding_dim
        elif emb_type == "vocab":
            self.module = VocabEmbedding(**kwargs)
            self.module.text_out_dim = self.embedding_dim
        elif emb_type == "projection":
            self.module = ProjectionEmbedding(**kwargs)
            self.module.text_out_dim = self.module.out_dim
        elif emb_type == "preextracted":
            self.module = PreExtractedEmbedding(**kwargs)
        elif emb_type == "bilstm":
            self.module = BiLSTMTextEmbedding(**kwargs)
        elif emb_type == "attention":
            self.module = AttentionTextEmbedding(**kwargs)
        elif emb_type == "mcan":
            self.module = SAEmbedding(**kwargs)
        elif emb_type == "torch":
            vocab_size = kwargs["vocab_size"]
            embedding_dim = kwargs["embedding_dim"]
            self.module = nn.Embedding(vocab_size, embedding_dim)
            self.module.text_out_dim = self.embedding_dim
        else:
            raise NotImplementedError("Unknown question embedding '%s'" % emb_type)

        self.text_out_dim = self.module.text_out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class VocabEmbedding(nn.Module):
    def __init__(self, embedding_dim, **vocab_params):
        super().__init__()
        self.vocab = Vocab(**vocab_params)
        self.module = self.vocab.get_embedding(
            nn.Embedding, embedding_dim=embedding_dim
        )

    def forward(self, x):
        return self.module(x)


class BiLSTMTextEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        num_layers,
        dropout,
        bidirectional=False,
        rnn_type="GRU",
    ):
        super().__init__()
        self.text_out_dim = hidden_dim
        self.bidirectional = bidirectional

        if rnn_type == "LSTM":
            rnn_cls = nn.LSTM
        elif rnn_type == "GRU":
            rnn_cls = nn.GRU

        self.recurrent_encoder = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.recurrent_encoder(x)
        # Return last state
        if self.bidirectional:
            return out[:, -1]

        forward_ = out[:, -1, : self.num_hid]
        backward = out[:, 0, self.num_hid :]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        output, _ = self.recurrent_encoder(x)
        return output


class PreExtractedEmbedding(nn.Module):
    def __init__(self, out_dim, base_path):
        super().__init__()
        self.text_out_dim = out_dim
        self.base_path = base_path
        self.cache = {}

    def forward(self, qids):
        embeddings = []
        for qid in qids:
            embeddings.append(self.get_item(qid))
        return torch.stack(embeddings, dim=0)

    @lru_cache(maxsize=5000)
    def get_item(self, qid):
        return np.load(os.path.join(self.base_path, str(qid.item()) + ".npy"))


class AttentionTextEmbedding(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers, dropout, **kwargs):
        super().__init__()

        self.text_out_dim = hidden_dim * kwargs["conv2_out"]

        bidirectional = kwargs.get("bidirectional", False)

        self.recurrent_unit = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        conv1_out = kwargs["conv1_out"]
        conv2_out = kwargs["conv2_out"]
        kernel_size = kwargs["kernel_size"]
        padding = kwargs["padding"]

        self.conv1 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        self.recurrent_unit.flatten_parameters()
        # self.recurrent_unit.flatten_parameters()
        lstm_out, _ = self.recurrent_unit(x)  # N * T * hidden_dim
        lstm_drop = self.dropout(lstm_out)  # N * T * hidden_dim
        lstm_reshape = lstm_drop.permute(0, 2, 1)  # N * hidden_dim * T

        qatt_conv1 = self.conv1(lstm_reshape)  # N x conv1_out x T
        qatt_relu = self.relu(qatt_conv1)
        qatt_conv2 = self.conv2(qatt_relu)  # N x conv2_out x T

        # Over last dim
        qtt_softmax = nn.functional.softmax(qatt_conv2, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature = torch.bmm(qtt_softmax, lstm_drop)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat = qtt_feature.view(batch_size, -1)

        return qtt_feature_concat


class ProjectionEmbedding(nn.Module):
    def __init__(self, module, in_dim, out_dim, **kwargs):
        super().__init__()
        if module == "linear":
            self.layers = nn.Linear(in_dim, out_dim)
            self.out_dim = out_dim
        elif module == "conv":
            last_out_channels = in_dim
            layers = []
            for conv in kwargs["convs"]:
                layers.append(nn.Conv1d(in_channels=last_out_channels, **conv))
                last_out_channels = conv["out_channels"]
            self.layers = nn.ModuleList(*layers)
            self.out_dim = last_out_channels
        else:
            raise TypeError(
                "Unknown module type for 'ProjectionEmbedding',"
                "use either 'linear' or 'conv'"
            )

    def forward(self, x):
        return self.layers(x)


class ImageFeatureEmbedding(nn.Module):
    """
    parameters:

    input:
    image_feat_variable: [batch_size, num_location, image_feat_dim]
    or a list of [num_location, image_feat_dim]
    when using adaptive number of objects
    question_embedding:[batch_size, txt_embeding_dim]

    output:
    image_embedding:[batch_size, image_feat_dim]


    """

    def __init__(self, img_dim, question_dim, **kwargs):
        super().__init__()

        self.image_attention_model = AttentionLayer(img_dim, question_dim, **kwargs)
        self.out_dim = self.image_attention_model.out_dim

    def forward(self, image_feat_variable, question_embedding, image_dims, extra=None):
        if extra is None:
            extra = {}
        # N x K x n_att
        attention = self.image_attention_model(
            image_feat_variable, question_embedding, image_dims
        )
        att_reshape = attention.permute(0, 2, 1)

        order_vectors = getattr(extra, "order_vectors", None)

        if order_vectors is not None:
            image_feat_variable = torch.cat(
                [image_feat_variable, order_vectors], dim=-1
            )
        tmp_embedding = torch.bmm(
            att_reshape, image_feat_variable
        )  # N x n_att x image_dim
        batch_size = att_reshape.size(0)
        image_embedding = tmp_embedding.view(batch_size, -1)

        return image_embedding, attention


class MultiHeadImageFeatureEmbedding(nn.Module):
    def __init__(self, img_dim, question_dim, **kwargs):
        super().__init__()
        self.module = nn.MultiheadAttention(
            embed_dim=question_dim, kdim=img_dim, vdim=img_dim, **kwargs
        )
        self.out_dim = question_dim

    def forward(self, image_feat_variable, question_embedding, image_dims, extra=None):
        if extra is None:
            extra = {}
        image_feat_variable = image_feat_variable.transpose(0, 1)
        question_embedding = question_embedding.unsqueeze(1).transpose(0, 1)
        output, weights = self.module(
            question_embedding, image_feat_variable, image_feat_variable
        )
        output = output.transpose(0, 1)

        return output.squeeze(), weights


class ImageFinetune(nn.Module):
    def __init__(self, in_dim, weights_file, bias_file):
        super().__init__()
        with PathManager.open(weights_file, "rb") as w:
            weights = pickle.load(w)
        with PathManager.open(bias_file, "rb") as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]

        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def forward(self, image):
        i2 = self.lc(image)
        i3 = nn.functional.relu(i2)
        return i3

class BertVisioLinguisticEmbeddings(BertEmbeddings):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        self.hidden_size= config.hidden_size

        self.position_embeddings_visual = nn.Embedding(
            config.max_img_len+1, config.hidden_size
        )
        self.position_embeddings = nn.Embedding(config.max_txt_len+1, config.hidden_size)
        self.type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size
        )

        self.words_projection1 = nn.Linear(300, config.hidden_size)
        self.words_projection2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)

    def initialize_visual_from_pretrained(self):

        self.position_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.position_embeddings.weight.data), requires_grad=True
        )

    def encode_text(
        self, input_embedding: Tensor
    ) -> Tensor:
        bs,length,feature_dim = input_embedding.shape
        word_embeddings = self.words_projection1(input_embedding.view(-1,feature_dim)).reshape(bs,length,-1)
        word_embeddings = self.words_projection1(word_embeddings.view(-1,self.hidden_size)).reshape(bs,length,-1)

        return word_embeddings

    def encode_image(
        self,
        visual_embeddings,
        visual_position_id
    ) -> Tensor:
    
        bs,num,c = visual_embeddings.shape

        visual_embeddings = self.projection(visual_embeddings.reshape(-1,c)).reshape(bs,num,-1)

        position_embeddings_visual = self.position_embeddings_visual(
                visual_position_id
            )
        # calculate visual embeddings
        v_embeddings = (
            visual_embeddings
            + position_embeddings_visual
        )
        return v_embeddings

    def forward(
        self,
        input_ids,
        type_tokens,
        visual_embeddings,
        visual_position_id
    ) -> Tensor:
        """
        input_ids = [batch_size, sequence_length]
        token_type_ids = [batch_size, sequence_length]
        visual_embedding = [batch_size, image_feature_length, image_feature_dim]
        image_text_alignment = [batch_size, image_feature_length, alignment_dim]
        """

        # text embeddings
        #text_embeddings = self.encode_text(input_ids)
        text_embeddings = self.encode_text(input_ids)

        # visual embeddings

        v_embeddings = self.encode_image(
                visual_embeddings,
        visual_position_id
            )

        token_type_embeddings = self.type_embeddings(type_tokens)

            # Concate the two:
        embeddings = torch.cat(
                (text_embeddings, v_embeddings), dim=1
            )  # concat the visual embeddings after the attentions

        embeddings += token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #print('embeddings',embeddings.shape)
        
        return embeddings

class BertVisioLinguisticEmbeddings_GloVe(BertEmbeddings):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        self.hidden_size= config.hidden_size

        self.position_embeddings_visual = nn.Embedding(
            config.max_img_len+1, config.hidden_size
        )
        self.position_embeddings = nn.Embedding(config.max_txt_len+1, config.hidden_size)
        self.type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size
        )

        self.words_projection1 = nn.Linear(300, config.hidden_size)
        self.words_projection2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)

    def initialize_visual_from_pretrained(self):

        self.position_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.position_embeddings.weight.data), requires_grad=True
        )

    def encode_text(
        self, input_embedding: Tensor
    ) -> Tensor:
        bs,length,feature_dim = input_embedding.shape
        word_embeddings = self.words_projection1(input_embedding.view(-1,feature_dim)).reshape(bs,length,-1)
        word_embeddings = self.words_projection1(word_embeddings.view(-1,self.hidden_size)).reshape(bs,length,-1)

        return word_embeddings

    def encode_image(
        self,
        visual_embeddings,
        visual_position_id
    ) -> Tensor:
        bs,num,c = visual_embeddings.shape

        visual_embeddings = self.projection(visual_embeddings.reshape(-1,c)).reshape(bs,num,-1)

        position_embeddings_visual = self.position_embeddings_visual(
                visual_position_id
            )
        # calculate visual embeddings
        v_embeddings = (
            visual_embeddings
            + position_embeddings_visual
        )
        return v_embeddings

    def forward(
        self,
        input_ids,
        type_tokens,
        visual_embeddings,
        visual_position_id
    ) -> Tensor:
        """
        input_ids = [batch_size, sequence_length]
        token_type_ids = [batch_size, sequence_length]
        visual_embedding = [batch_size, image_feature_length, image_feature_dim]
        image_text_alignment = [batch_size, image_feature_length, alignment_dim]
        """

        # text embeddings
        #text_embeddings = self.encode_text(input_ids)
        text_embeddings = self.encode_text(input_ids)

        # visual embeddings

        v_embeddings = self.encode_image(
                visual_embeddings,
        visual_position_id
            )

        token_type_embeddings = self.type_embeddings(type_tokens)

            # Concate the two:
        embeddings = torch.cat(
                (text_embeddings, v_embeddings), dim=1
            )  # concat the visual embeddings after the attentions
        embeddings += token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #print('embeddings',embeddings.shape)
        
        return embeddings


class SituationGraphEmbeddings(BertEmbeddings):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.channel = config.frame_feature_channel
        
        self.max_act = config.max_act
        self.max_frame = config.max_frame
        self.max_rel = config.max_rel

        self.symbolic = config.symbolic
        self.semantic = config.semantic
        self.visual = config.visual

        _max_rel = config.max_rel + 3
        _max_frame = config.max_frame + 1
        self.type_embedding = nn.Embedding(5, config.hidden_size)
        self.trip_embedding = nn.Embedding(_max_rel, config.hidden_size)
        self.situation_embedding = nn.Embedding(_max_frame, config.hidden_size)
        self.hyper_embedding = nn.Embedding(3, config.hidden_size)

        self.frame_conv = nn.Sequential(
             nn.Conv2d(self.channel,self.channel,3,2),
             nn.ReLU(inplace=True),
             nn.BatchNorm2d(self.channel),
             nn.Conv2d(self.channel,self.channel,3,2),
             nn.ReLU(inplace=True),
             nn.BatchNorm2d(self.channel)
            )

        self.frame_projection = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(config.frame_embedding_dim, config.hidden_size),
                nn.ReLU(inplace=True),
            )

        # graph_representation
        self.symbolic_only = self.symbolic & (~self.semantic)
        self.semantic_only = self.semantic & (~self.symbolic)
        self.fusion = self.semantic & self.symbolic

        if self.symbolic_only:
            self.symbolic_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        elif self.semantic_only:
            self.senmatic_embedding = nn.Linear(300, config.hidden_size)

        elif self.fusion:
            self.symbolic_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

            self.fusion_embedding = self.obj_downsample = nn.Sequential(
                    nn.Dropout(p=0.1),
                    nn.Linear(300+config.hidden_size, config.hidden_size),
                    nn.ReLU(inplace=True),
                )

        if self.visual:

            self.obj_projection = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(config.obj_feature_size, config.hidden_size),
                nn.ReLU(inplace=True),
            )

            obj_fusion_dim = config.hidden_size
            if self.symbolic:
                obj_fusion_dim += config.hidden_size
                
            if self.semantic:
                obj_fusion_dim += 300

            self.obj_fusion = nn.Sequential(
                    nn.Dropout(p=0.1),
                    nn.Linear(obj_fusion_dim, config.hidden_size),
                    nn.ReLU(inplace=True),
                )

    def forward(
        self,
        act_id, obj1_id, rel_id, obj2_id, special_id,
        frame_feature, obj1_feature, obj2_feature,
        special_semantic, act_semantic, obj1_semantic, rel_semantic, obj2_semantic,
        type_token, hyper_token, triplet_token, situation_token
    ):

        #print(act_id.shape) # (B, max_frame, max_act)
        b,t,c,w,h = frame_feature.shape
        #frame_feature = frame_feature.reshape(-1,c,w,h)
        #frame = self.frame_conv(frame_feature)
        #frame = frame.reshape(b,t,-1)
        #frame = self.frame_projection(frame)

        type_embed = self.type_embedding(type_token)
        hyper_embed = self.hyper_embedding(hyper_token)
        trip_embed = self.trip_embedding(triplet_token)
        situ_embed = self.situation_embedding(situation_token)

        if self.symbolic_only:
            #spe = self.symbolic_embedding(special_id)
            act = self.symbolic_embedding(act_id) # (B, max_frame, max_act, hidden_size)
            rel = self.symbolic_embedding(rel_id)
            obj1 = self.symbolic_embedding(obj1_id)
            obj2 = self.symbolic_embedding(obj2_id)

            if self.visual:
            #b, t, _, _ = obj1_feature.shape # [b, max_frame, max_rel ,2048]
                obj1_feature = self.obj_projection(obj1_feature) # [b, max_frame, max_rel, 256]
                #print('obj1_feature',obj1_feature.shape)
                obj2_feature = self.obj_projection(obj2_feature)
                obj1 = torch.cat([obj1,obj1_feature],dim=-1)
                obj2 = torch.cat([obj2,obj2_feature],dim=-1)
                #print('_obj2',obj2.shape)
                obj1 = self.obj_fusion(obj1)
                obj2 = self.obj_fusion(obj2)
            #print('obj2',obj2.shape)

        elif self.semantic_only:
            #print('act_semantic',act_semantic.shape)
            act = self.senmatic_embedding(act_semantic) # (B, max_frame, max_act, hidden_size)
            rel = self.senmatic_embedding(rel_semantic)
            #print('special_semantic',special_semantic.shape)
            #spe = self.senmatic_embedding(special_semantic)

            if self.visual:
                obj1_feature = self.obj_projection(obj1_feature) # [b, max_frame, max_rel, 256]
                obj2_feature = self.obj_projection(obj2_feature)
                obj1 = torch.cat([obj1_semantic,obj1_feature],dim=-1)
                obj2 = torch.cat([obj2_semantic,obj2_feature],dim=-1)
                obj1 = self.obj_fusion(obj1)
                obj2 = self.obj_fusion(obj2)
            else:
                obj1 = self.senmatic_embedding(obj1_semantic)
                obj2 = self.senmatic_embedding(obj2_semantic)

        elif self.fusion:
            #spe = self.symbolic_embedding(special_id)
            act = self.symbolic_embedding(act_id) # (B, max_frame, max_act, hidden_size)
            rel = self.symbolic_embedding(rel_id)
            obj1 = self.symbolic_embedding(obj1_id)
            obj2 = self.symbolic_embedding(obj2_id)

            #spe = torch.cat([special_semantic,spe],dim=-1)
            act = torch.cat([act_semantic,act],dim=-1)
            rel = torch.cat([rel_semantic,rel],dim=-1)

            #spe = self.fusion_embedding(spe)
            act = self.fusion_embedding(act)
            rel = self.fusion_embedding(rel)
            if self.visual:
                obj1_feature = self.obj_projection(obj1_feature) # [b, max_frame, max_rel, 256]
                obj2_feature = self.obj_projection(obj2_feature)
                obj1 = torch.cat([obj1_semantic,obj1, obj1_feature],dim=-1)
                obj2 = torch.cat([obj2_semantic,obj2, obj2_feature],dim=-1)

                obj1 = self.obj_fusion(obj1)
                obj2 = self.obj_fusion(obj2)
            else:
                obj1 = torch.cat([obj1_semantic, obj1],dim=-1)
                obj2 = torch.cat([obj2_semantic, obj2],dim=-1)
                obj1 = self.fusion_embedding(obj1)
                obj2 = self.fusion_embedding(obj2)

        b, t, _, d = act.shape
        #frame = frame.unsqueeze(-2)
        #feature_embed = torch.stack([spe,obj1,rel,obj2], dim=-2).reshape(b, t, 4 * self.max_rel, d)
        #feature_embed = torch.cat([act,feature_embed,frame], dim=-2).reshape(b,-1,d)
        feature_embed = torch.stack([obj1,rel,obj2], dim=-2).reshape(b, t, 3 * self.max_rel, d)
        feature_embed = torch.cat([act,feature_embed], dim=-2).reshape(b,-1,d)

        #print('feature_embed',feature_embed.shape)
        
        embed = feature_embed + 1*(type_embed + trip_embed + 5*situ_embed + 1*hyper_embed)
        embed = self.LayerNorm(embed)
        embed = self.dropout(embed)

        #print('embed',embed.shape) #[b, 560, 256] 560 = (2+4x8+1)X16
        return embed


class SAEmbedding(nn.Module):
    """Encoder block implementation in MCAN https://arxiv.org/abs/1906.10770
    """

    def __init__(self, hidden_dim: int, embedding_dim: int, **kwargs):
        super().__init__()
        num_attn = kwargs["num_attn"]
        num_layers = kwargs["num_layers"]
        dropout = kwargs.get("dropout", 0.1)
        num_attn_pool = kwargs.get("num_attn_pool", 1)
        num_feat = kwargs.get("num_feat", -1)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.self_attns = nn.ModuleList(
            [SelfAttention(hidden_dim, num_attn, dropout) for _ in range(num_layers)]
        )
        self.attn_pool = None
        self.num_feat = num_feat
        self.text_out_dim = hidden_dim
        if num_attn_pool > 0:
            self.attn_pool = AttnPool1d(hidden_dim, num_feat * num_attn_pool)
            self.text_out_dim = hidden_dim * num_attn_pool

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        out, (h, c) = self.lstm(x)
        for self_attn in self.self_attns:
            out = self_attn(out, mask)

        vec = h.transpose(0, 1).contiguous().view(b, 1, -1)
        if self.attn_pool:
            vec = self.attn_pool(out, out, mask).view(b, self.num_feat, -1)

        return out, vec


class SGAEmbedding(nn.Module):
    """Decoder block implementation in MCAN https://arxiv.org/abs/1906.10770
    """

    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__()
        num_attn = kwargs["num_attn"]
        num_layers = kwargs["num_layers"]
        dropout = kwargs.get("dropout", 0.1)
        hidden_dim = kwargs.get("hidden_dim", 512)

        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.self_guided_attns = nn.ModuleList(
            [
                SelfGuidedAttention(hidden_dim, num_attn, dropout)
                for _ in range(num_layers)
            ]
        )
        self.out_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.view(b, c, -1).transpose(1, 2).contiguous()  # b x (h*w) x c

        x = self.linear(x)

        for self_guided_attn in self.self_guided_attns:
            x = self_guided_attn(x, y, x_mask, y_mask)

        return x


class CBNEmbedding(nn.Module):
    """MoVie bottleneck layers from https://arxiv.org/abs/2004.11883
    """

    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__()
        cond_dim = kwargs["cond_dim"]
        num_layers = kwargs["cbn_num_layers"]
        compressed = kwargs.get("compressed", True)
        use_se = kwargs.get("use_se", True)

        self.out_dim = 1024
        self.layer_norm = nn.LayerNorm(self.out_dim)
        cbns = []
        for i in range(num_layers):
            if embedding_dim != self.out_dim:
                downsample = nn.Conv2d(
                    embedding_dim, self.out_dim, kernel_size=1, stride=1, bias=False
                )
                cbns.append(
                    MovieBottleneck(
                        embedding_dim,
                        self.out_dim // 4,
                        cond_dim,
                        downsample=downsample,
                        compressed=compressed,
                        use_se=use_se,
                    )
                )
            else:
                cbns.append(
                    MovieBottleneck(
                        embedding_dim,
                        self.out_dim // 4,
                        cond_dim,
                        compressed=compressed,
                        use_se=use_se,
                    )
                )
            embedding_dim = self.out_dim
        self.cbns = nn.ModuleList(cbns)
        self._init_layers()

    def _init_layers(self) -> None:
        for cbn in self.cbns:
            cbn.init_layers()

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        for cbn in self.cbns:
            x, _ = cbn(x, v)

        x = self.layer_norm(
            nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(3).squeeze(2)
        )

        return x


class TwoBranchEmbedding(nn.Module):
    """Attach MoVie into MCAN model as a counting module in
    https://arxiv.org/abs/2004.11883
    """

    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__()
        hidden_dim = kwargs.get("hidden_dim", 512)
        self.sga = SGAEmbedding(embedding_dim, **kwargs)
        self.sga_pool = AttnPool1d(hidden_dim, 1)
        self.cbn = CBNEmbedding(embedding_dim, **kwargs)
        self.out_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_sga = self.sga(x, y, x_mask, y_mask)
        x_sga = self.sga_pool(x_sga, x_sga, x_mask).squeeze(1)
        x_cbn = self.cbn(x, v)

        return x_sga, x_cbn
