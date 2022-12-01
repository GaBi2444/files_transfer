
from copy import deepcopy

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ClassifierLayer, ConvNet
from torch import nn
import torchvision.models as model

_TEMPLATES = {
    "question_vocab_size": "{}_text_vocab_size",
    "number_of_answers": "{}_num_final_outputs",
}

_CONSTANTS = {"hidden_state_warning": "hidden state (final) should have 1st dim as 2"}

@registry.register_model("2dcnn_mlp")
class _2DCNN_MLP(BaseModel):
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
        return "configs/models/2dcnn_mlp/defaults.yaml"

    def build(self):
        assert len(self._datasets) > 0

        resnet = model.resnext50_32x4d(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules).eval()

        # As we generate output dim dynamically, we need to copy the config
        # to update it
        classifier_config = deepcopy(self.config.classifier)
        classifier_config.params.out_dim = 4
        classifier_config.params.in_dim = 2048*classifier_config.params.video_len + 300
        self.classifier = ClassifierLayer(
            classifier_config.type, **classifier_config.params
        )
        

    def forward(self, sample_list):

        #q = self.text_embedding(sample_list.question)
        #choice = self.text_embedding(sample_list.choice)
        bs,num, c , w, h = sample_list.video.shape
        text = torch.mean(sample_list.text,dim=1)

        self.feature_extractor.eval()
        with torch.no_grad():
            video_features = self.feature_extractor(sample_list.video.view(-1,c,w,h).contiguous()).reshape(bs,-1)

        fused = torch.cat([video_features,text], dim=-1)
        scores = self.classifier(fused)

        return {"scores": scores}
