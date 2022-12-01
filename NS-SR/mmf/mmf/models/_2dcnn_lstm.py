from copy import deepcopy

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ClassifierLayer, ConvNet, Flatten,Conv3DNet
from torch import nn
import torchvision.models as model

_TEMPLATES = {
    "question_vocab_size": "{}_text_vocab_size",
    "number_of_answers": "{}_num_final_outputs",
}

_CONSTANTS = {"hidden_state_warning": "hidden state (final) should have 1st dim as 2"}


@registry.register_model("2dcnn_lstm")
class _2DCNNLSTM(BaseModel):
    """CNNLSTM is a simple model for vision and language tasks. CNNLSTM is supposed
    to acts as a baseline to test out your stuff without any complex functionality.
    Passes image through a CNN, and text through an LSTM and fuses them using
    concatenation. Then, it finally passes the fused representation from a MLP to
    generate scores for each of the possible answers.
    Args:
        config (DictConfig): Configuration node containing all of the necessary
                             config required to initialize CNNLSTM.
    Inputs: sample_list (SampleList)
        - **sample_list** should contain image attribute for image, text for
          question split into word indices, targets for answer scores
    """
    def __init__(self, config):
        super().__init__(config)
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/2dcnn_lstm/defaults.yaml"

    def build(self):
        assert len(self._datasets) > 0

        self.lstm = nn.LSTM(**self.config.lstm)

        #self.visual_encoder = VideoEncoderCNN(**self.config.video_encoder)
        resnet = model.resnext50_32x4d(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        # As we generate output dim dynamically, we need to copy the config
        # to update it
        classifier_config = deepcopy(self.config.classifier)
        classifier_config.params.out_dim = 4
        classifier_config.params.in_dim = 2048*classifier_config.params.video_len + 2*self.config.lstm.hidden_size

        self.classifier = ClassifierLayer(
            classifier_config.type, **classifier_config.params
        )
        
    def forward(self, sample_list):
        self.lstm.flatten_parameters()
        text = sample_list.text
        # Get (h_n, c_n), last hidden and cell state
        _, hidden = self.lstm(text)
        # X x B x H => B x X x H where X = num_layers * num_directions
        hidden= hidden[0].transpose(0, 1)
        # X should be 2 so we can merge in that dimension
        hidden = torch.cat([hidden[:, 0, :], hidden[:, 1, :]], dim=-1)

        bs,num, c , w, h = sample_list.video.shape
        self.feature_extractor.eval()
        with torch.no_grad():
            video_features = self.feature_extractor(sample_list.video.view(-1,c,w,h).contiguous()).reshape(bs,-1)

        #print('video',video_features.shape)
        # Fuse into single dimension
        fused = torch.cat([video_features, hidden], dim=-1)
        scores = self.classifier(fused)

        return {"scores": scores}
