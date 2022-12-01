import json
import logging
import math
import os
import zipfile
from collections import Counter

from mmf.common.constants import CLEVR_DOWNLOAD_URL
from mmf.common.registry import registry
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
from mmf.datasets.builders.clevrer.dataset import CLEVRERDataset
from mmf.utils.download import download
from mmf.utils.general import get_mmf_root


logger = logging.getLogger(__name__)


@registry.register_builder("clevrer")
class CLEVRERBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__("clevrer")
        self.dataset_class = CLEVRERDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/clevrer/defaults.yaml"

    def build(self, config, dataset_type):
        pass
        # plesae manually download dataset in http://clevrer.csail.mit.edu/

    def load(self, config, dataset_type, *args, **kwargs):
        self.dataset = CLEVRERDataset(config, dataset_type)
        return self.dataset

    def update_registry_for_model(self, config):
        registry.register(
            self.dataset_name + "_text_vocab_size",
            self.dataset.text_processor.get_vocab_size(),
        )
        registry.register(
            self.dataset_name + "_num_final_outputs",
            self.dataset.answer_processor.get_vocab_size(),
        )
