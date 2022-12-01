# Copyright (c) Facebook, Inc. and its affiliates.

import collections

import torch
import copy
from mmf.common.registry import registry
from mmf.datasets.processors.processors import BaseProcessor
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.ops import nms, roi_align, roi_pool


@registry.register_processor("torchvision_transforms")
class TorchvisionTransforms(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        transform_params = config.transforms
        assert OmegaConf.is_dict(transform_params) or OmegaConf.is_list(
            transform_params
        )
        if OmegaConf.is_dict(transform_params):
            transform_params = [transform_params]

        transforms_list = []

        for param in transform_params:
            if OmegaConf.is_dict(param):
                # This will throw config error if missing
                transform_type = param.type
                transform_param = param.get("params", OmegaConf.create({}))
            else:
                assert isinstance(param, str), (
                    "Each transform should either be str or dict containing "
                    + "type and params"
                )
                transform_type = param
                transform_param = OmegaConf.create([])

            transform = getattr(transforms, transform_type, None)
            # If torchvision doesn't contain this, check our registry if we
            # implemented a custom transform as processor
            if transform is None:
                transform = registry.get_processor_class(transform_type)
            assert (
                transform is not None
            ), f"torchvision.transforms has no transform {transform_type}"

            # https://github.com/omry/omegaconf/issues/248
            transform_param = OmegaConf.to_container(transform_param)
            # If a dict, it will be passed as **kwargs, else a list is *args
            if isinstance(transform_param, collections.abc.Mapping):
                transform_object = transform(**transform_param)
            else:
                transform_object = transform(*transform_param)

            transforms_list.append(transform_object)

        self.transform = transforms.Compose(transforms_list)

    def __call__(self, x):
        # Support both dict and normal mode
        if isinstance(x, collections.abc.Mapping):
            x = x["image"]
            return {"image": self.transform(x)}
        else:
            return self.transform(x)


@registry.register_processor("GrayScaleTo3Channels")
class GrayScaleTo3Channels(BaseProcessor):
    def __init__(self, *args, **kwargs):
        return

    def __call__(self, x):
        if isinstance(x, collections.abc.Mapping):
            x = x["image"]
            return {"image": self.transform(x)}
        else:
            return self.transform(x)

    def transform(self, x):
        assert isinstance(x, torch.Tensor)
        # Handle grayscale, tile 3 times
        if x.size(0) == 1:
            x = torch.cat([x] * 3, dim=0)
        return x

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transform = lambda s: transforms.Compose(
            [transforms.Resize(s), transforms.RandomCrop(s), transforms.ToTensor(), normalize])
test_transform = lambda s: transforms.Compose(
            [transforms.Resize(s), transforms.CenterCrop(s), transforms.ToTensor(), normalize])

@registry.register_processor("image_resize_crop")
class Image_Resize_Crop(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        self.transform_type = config.transform_type
        self.bbox_size = (config.resize_width,config.resize_height)
        self.train_transform = train_transform(self.bbox_size)
        self.test_transform = test_transform(self.bbox_size)

    def __call__(self, x):
        if self.transform_type == 'train' or 'val':
            return self.train_transform(x)
        if self.transform_type == 'test':
            return self.test_transform(x)


@registry.register_processor("visual_feature_cropper")
class VisualFeatureCropper(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        self.output_size = config.output_size
        self.spatial_scale = config.spatial_scale
        self.sampling_ratio = config.sampling_ratio
        self.aligned = config.aligned
        self.visual_feature_size = config.visual_feature_size
        self.max_obj = config.max_obj

        self.padding = torch.zeros([self.max_obj, self.visual_feature_size], dtype=torch.float32)

    def __call__(self, labels, bboxes, select_keyframe, base_feature):
        frame_id = sorted(select_keyframe.keys())
        features = []
        for i, (lab, _id)  in enumerate(zip(labels,frame_id)):
            #print(lab,_id)
            #print(bboxes[_id])
            if select_keyframe[_id] == 0: # unseen
                features.append(copy.deepcopy(self.padding))
            else:
                feature = []
                base = base_feature[i].unsqueeze(0) # (1,512,512)
                for l in lab:
                    if l not in bboxes[_id]:
                        feature.append(copy.deepcopy(self.padding[0]))
                    else:
                        box = torch.tensor(bboxes[_id][l])
                        box = torch.cat([torch.tensor([0.]),box])
                        box = box.unsqueeze(0)
                        pooled_features = roi_align(base, box, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned)
                        pooled_features = pooled_features.reshape(-1)
                        feature.append(pooled_features)
                feature = torch.stack(feature)
                features.append(feature)
        features = torch.stack(features)

        return features

