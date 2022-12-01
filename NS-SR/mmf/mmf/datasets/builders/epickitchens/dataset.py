import json
import os

import numpy as np
import torch
from PIL import Image

import csv
import pickle
import random
from collections import defaultdict
import torchvision.transforms as transforms
from torch.utils import data
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from mmf.common.registry import registry
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.utils.distributed import is_master, synchronize
from mmf.utils.general import get_mmf_root
from mmf.utils.text import VocabFromText, tokenize



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transform = lambda s: transforms.Compose(
    [transforms.Resize(s), transforms.RandomCrop(s), transforms.ToTensor(), normalize])
test_transform = lambda s: transforms.Compose(
    [transforms.Resize(s), transforms.CenterCrop(s), transforms.ToTensor(), normalize])


class EpicKitchensDataset(BaseDataset):

    def __init__(self, config, dataset_type, *args, **kwargs):
        super().__init__('epickitchens', config, dataset_type)

        self.max_img_seq_len = self.config.max_img_seq_len
        self.max_txt_seq_len = self.config.max_txt_seq_len
        self.include_whole_img = self.config.include_whole_img
        self.img_root = self.config.img_root
        self.bbox_transform = transforms.Lambda(lambda x: x)

        self.actions = self.process_actions(
            torch.load(self.config.annotation.actions)['actions'])
        with open(self.config.annotation.objects) as f:
            self.objects = self.process_objects(csv.DictReader(f))
        with open(self.config.annotation.video_info) as f:
            self.video_info = self.process_video_info(csv.DictReader(f))
        split_data = torch.load(self.config.annotation.splits)
        self.split_data = split_data
        vocab = split_data['vocab']
        self.vocab = vocab

        if dataset_type == 'train':
            split_data = split_data['train_action_uids']
        elif dataset_type == 'val' or dataset_type == 'test':
            split_data = split_data['test_action_uids']

        self.action_uids = set(e for r in split_data for c in r for e in c)
        self.sel_action_uids = set(
            e for i, r in enumerate(split_data) for j, c in enumerate(r) for e in c if
            i in vocab['sel_verb_idxs'] and j in vocab['sel_noun_idxs'])
        self.frames = self.get_frames()

    def get_frames(self):
        frames = []
        for vid_id, d in self.actions.items():
            for act_id, row in enumerate(d):
                if int(row['uid']) in self.action_uids:
                    i = int(row['start_frame'])
                    j = int(row['stop_frame'])
                    frames.extend([(vid_id, act_id, n) for n in range(i, j) if self.objects[vid_id][n]])
        return frames[:1792]

    def get_raw_image(self, index, bbox=False):
        vid_id, act_id, frame_id = self.frames[index]
        participant_id = vid_id.split('_')[0]
        img_path = os.path.join(self.img_root, participant_id, vid_id, f'frame_{frame_id:010d}.jpg')
        # To use high resolution images, the videos need to be downloaded first
        img = default_loader(img_path)  # this loads a smaller version of the image
        if bbox:
            img_bboxes = []
            objects = self.objects[vid_id][frame_id]
            orig_w, orig_h = self.video_info[vid_id]['res']
            img_w, img_h = img.size
            for obj in objects:
                for t, l, h, w in obj['bbox']:
                    h_scale = img_h / orig_h
                    w_scale = img_w / orig_w
                    t *= h_scale
                    h *= h_scale
                    l *= w_scale
                    w *= w_scale
                    if h < 10 or w < 10: continue  # too thin or narrow? do not add bbox
                    bbox = [int(l), int(t), int(l + w), int(t + h)]
                    img_bboxes.append(img.crop(bbox))
            return img, img_bboxes
        return img

    @staticmethod
    def process_video_info(d):
        ret = {}
        for row in d:
            ret[row['video']] = {'res': list(map(int, row['resolution'].split('x'))), 'fps': row['fps'],
                                 'dur': float(row['duration'])}
        return ret

    @staticmethod
    def process_actions(d):
        ret = defaultdict(list)
        for row in d:
            ret[row['video_id']].append(row)
        return ret

    @staticmethod
    def process_objects(d):
        ret = defaultdict(lambda: defaultdict(list))
        for row in d:
            obj = row
            obj['bbox'] = obj['bounding_boxes']
            if obj['bbox']: obj['bbox'] = eval(obj['bbox'])
            if len(obj['bbox']):
                ret[row['video_id']][int(row['frame'])].append(obj)
        return ret

    def pad_text(self, txt, l=None):
        """
        :param txt: list of text tokens
        """
        l = l or self.max_txt_seq_len
        txt = txt[:l]
        return (txt + [self.tokenizer.pad_token for i in range(l - len(txt))]), len(txt)

    def pad_imgs(self, imgs):
        ret = torch.zeros((self.max_img_seq_len, *tuple(imgs.shape[1:]))).to(dtype=imgs.dtype, device=imgs.device)
        k = min(len(imgs), self.max_img_seq_len)
        ret[:k] = imgs[:k]
        return ret

    def __len__(self):
        nums = len(self.frames)#change size /50
        #if self.split == "train":
        #    nums =  len(self.frames) // 5 # for training
        print('The size of ' + self.dataset_type + ' set:', nums)
        return nums  # for testing or training

    def __getitem__(self, i, pad=True, add_special_toks=True):
        current_sample = Sample()
        vid_id, act_id, frame_id = self.frames[i]
        action = self.actions[vid_id][act_id]
        objects = self.objects[vid_id][frame_id]
        participant_id = vid_id.split('_')[0]

        # Images
        img_path = os.path.join(self.img_root, participant_id, vid_id, f'frame_{frame_id:010d}.jpg')
        img = np.true_divide(Image.open(img_path).convert("RGB"), 255)
        img = img.astype(np.float32)
        
        current_sample.image = torch.from_numpy(img.transpose(2, 0, 1))
        processed  = self.text_processor({"text": action['narration']})
        current_sample.token = processed["text"]
        current_sample.text = processed["input_ids"]
        current_sample.targets = processed["input_mask"]

        return current_sample

