
from copy import deepcopy

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ClassifierLayer, ConvNet, Flatten,Conv3DNet
from torch import nn


_TEMPLATES = {
    "question_vocab_size": "{}_text_vocab_size",
    "number_of_answers": "{}_num_final_outputs",
}

_CONSTANTS = {"hidden_state_warning": "hidden state (final) should have 1st dim as 2"}


@registry.register_model("lstm")
class LSTM(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/lstm/defaults.yaml"

    def build(self):
        assert len(self._datasets) > 0
        #self.text_embedding = nn.Embedding(
        #    self.num_question_choices, self.config.text_embedding.embedding_dim
        #)
        self.lstm = nn.LSTM(**self.config.lstm)
        classifier_config = self.config.classifier
        self.classifier = ClassifierLayer(
            classifier_config.type, **classifier_config.params
        )

    def forward(self, sample_list):
        self.lstm.flatten_parameters()

        input = sample_list.text
        _, hidden = self.lstm(input)
        # X x B x H => B x X x H where X = num_layers * num_directions
        hidden = hidden[0].transpose(0, 1)
        # X should be 2 so we can merge in that dimension
        #assert hidden_c.size(1) == 2, _CONSTANTS["hidden_state_warning"]
        hidden = torch.cat([hidden[:, 0, :], hidden[:, 1, :]], dim=-1)
        scores = self.classifier(hidden)
        softmax = nn.Softmax(dim=-1)
        scores = softmax(scores)

        return {"scores": scores}
