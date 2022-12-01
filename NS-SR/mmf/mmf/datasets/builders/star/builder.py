import json
import logging
import math
import os
import zipfile
from collections import Counter

from mmf.common.registry import registry
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
from mmf.datasets.builders.star.dataset import STARDataset
from mmf.utils.download import download
from mmf.utils.general import get_mmf_root


logger = logging.getLogger(__name__)


@registry.register_builder("star")
class STARBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__("star")
        self.dataset_class = STARDataset
        self.dataset_name = "star"

    @classmethod
    def config_path(cls):
        return "configs/datasets/star/defaults.yaml"

    def build(self, config, dataset_type):
        pass
        # TODO: download dataset automatically

    def load(self, config, dataset_type, *args, **kwargs):
        self.dataset = self.dataset_class(config, dataset_type)
        return self.dataset

    def update_registry_for_model(self, config):
        pass
