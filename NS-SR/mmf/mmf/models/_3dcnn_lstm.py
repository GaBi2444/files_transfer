
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


@registry.register_model("3dcnn_lstm")
class _3DCNNLSTM(BaseModel):
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
        return "configs/models/cnn_lstm_for_vqa/defaults.yaml"

    def build(self):
        assert len(self._datasets) > 0
        # self.num_question_choices = registry.get(
        #     _TEMPLATES["question_vocab_size"].format(self._datasets[0])
        # )
        # self.num_answer_choices = registry.get(
        #     _TEMPLATES["number_of_answers"].format(self._datasets[0])
        # )

        #self.text_embedding = nn.Embedding(
        #    self.num_question_choices, self.config.text_embedding.embedding_dim
        #)
        self.lstm_q = nn.LSTM(**self.config.lstm)
        self.lstm_c = nn.LSTM(**self.config.lstm)

        #self.visual_encoder = VideoEncoderCNN(**self.config.video_encoder)
        resnet = model.resnext50_32x4d(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.image_feature_extractor = nn.Sequential(*modules).eval()

        # As we generate output dim dynamically, we need to copy the config
        # to update it
        classifier_config = deepcopy(self.config.classifier)
        classifier_config.params.out_dim = 2
        classifier_config.params.in_dim = 2048*classifier_config.params.video_len + 4*self.config.lstm.hidden_size

        self.classifier = ClassifierLayer(
            classifier_config.type, **classifier_config.params
        )
        

    def forward(self, sample_list):
        self.lstm_q.flatten_parameters()
        self.lstm_c.flatten_parameters()

        #text_input = torch.cat((sample_list.question,sample_list.choice),dim=1)
        #q = self.text_embedding(sample_list.question)
        #c = self.text_embedding(sample_list.choice)
        # Get (h_n, c_n), last hidden and cell state
        _, hidden_q = self.lstm_q(sample_list.question)
        _, hidden_c = self.lstm_c(sample_list.choice)
        # X x B x H => B x X x H where X = num_layers * num_directions
        hidden_q= hidden_q[0].transpose(0, 1)
        hidden_c= hidden_c[0].transpose(0, 1)
        # X should be 2 so we can merge in that dimension
        assert hidden_q.size(1) == 2, _CONSTANTS["hidden_state_warning"]
        hidden_q = torch.cat([hidden_q[:, 0, :], hidden_q[:, 1, :]], dim=-1)
        hidden_c = torch.cat([hidden_c[:, 0, :], hidden_c[:, 1, :]], dim=-1)
        hidden = torch.cat([hidden_q,hidden_c],dim=-1)

        bs,num, c , w, h = sample_list.video.shape
        
        with torch.no_grad():
            video_features = self.image_feature_extractor(sample_list.video.view(-1,c,w,h).contiguous()).reshape(bs,-1)

        fused = torch.cat([video_features, hidden], dim=-1)
        scores = self.classifier(fused)

        return {"scores": scores}
