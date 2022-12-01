from mmf.common.registry import registry
from mmf.datasets.builders.star.builder import STARBuilder
from mmf.datasets.builders.star.dataset_nssr import STARDataset_NSSR

@registry.register_builder("star_nssr")
class MaskedVQA2Builder(STARBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "star_nssr"
        self.dataset_class = STARDataset_NSSR

    @classmethod
    def config_path(cls):
        return "configs/datasets/star/nssr.yaml"