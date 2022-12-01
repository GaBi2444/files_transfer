import json
import logging
import math
import os
import zipfile
from collections import Counter

from mmf.common.constants import CLEVR_DOWNLOAD_URL
from mmf.common.registry import registry
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
from mmf.datasets.builders.epickitchens.dataset import EpicKitchensDataset
from mmf.utils.download import download
from mmf.utils.general import get_mmf_root

logger = logging.getLogger(__name__)


@registry.register_builder("epickitchens")
class EpicKitchensBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__("epickitchens")
        self.dataset_name = 'epickitchens'
        self.dataset_class = EpicKitchensDataset
        # register datasetname to mmf

    @classmethod
    def config_path(cls):
        return "configs/datasets/epickitchens/defaults.yaml"

    def build(self,config,dataset_type):
        pass
        # we have already download dataset mannually

    def load(self, config, dataset_type, *args, **kwargs):
        self.dataset = EpicKitchensDataset(config, dataset_type)
        return self.dataset

    def update_registry_for_model(self, config):
        # we define the processor in dataset.py
        registry.register(
            self.dataset_name + "_text_vocab_size",
            self.dataset.text_processor.get_vocab_size(),
        )
#mmf_run config=projects/others/cnn_lstm/epickitchens/defaults.yaml datasets=epickitchen model=cnn-lstm run_type=train_val