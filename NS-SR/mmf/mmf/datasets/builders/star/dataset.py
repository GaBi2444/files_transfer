import os
import json
import torch
import pickle
import random
import copy

import logging
import numpy as np

from mmf.common.registry import registry
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.utils.distributed import is_master, synchronize
from mmf.utils.general import get_mmf_root
from mmf.utils.text import tokenize
from PIL import Image

logger = logging.getLogger(__name__)

_QTYPE = ['Interaction', 'Sequence', 'Prediction', 'Feasibility']

_CONSTANTS = {
    "OBJ_NUM": 37, 
    "ACT_NUM": 111,
    "REL_NUM": 24,
    "dataset_key": 'star'
}

_TEMPLATES = {
    "data_folder_missing_error": "Data folder {} for STAR is not present.",
}


class STARDataset(BaseDataset):

    def __init__(self, config, dataset_type, data_folder=None, *args, **kwargs):
        super().__init__(_CONSTANTS["dataset_key"], config, dataset_type)

        self._dataset_type = dataset_type
        self._data_folder = data_folder
        self._data_dir = os.path.join(get_mmf_root(), config.data_dir)
        if not self._data_folder:
            self._data_folder = os.path.join(self._data_dir, config.data_folder)

        if not os.path.exists(self._data_folder):
            raise RuntimeError(
                _TEMPLATES["data_folder_missing_error"].format(self._data_folder)
            )

        if config.data_folder in os.listdir(self._data_folder):
            self._data_folder = os.path.join(self._data_folder, config.data_folder)

        self.qa_folder = config.qa_folder
        self.anno_folder = config.anno_folder
        self.frame_folder = config.frame_folder

        self.train_graph = config.train_graph
        self.val_graph = config.val_graph
        self.graph_type = config.graph_type

        qtype = config.qtype.title()
        if qtype not in _QTYPE:
            qtype = 'STAR'
        self.qa_json = qtype + '_' + dataset_type + '.json'

        self.load_json()

    def load_json(self):

        with open(os.path.join(self._data_folder,self.qa_folder,self.qa_json)) as f:
            self.sr_qa = json.load(f)

    def __len__(self):
        return len(self.sr_qa) # situation reasoning QA

    def __getitem__(self, idx):

        qa = self.sr_qa[idx]
        
        current_sample = Sample()
        question = qa["question"]
        choices = qa['choices']
        video_id, question_id = qa['video_id'], qa['question_id']
        start_time, end_time = qa['start'], qa['end']
        #situations = qa['situations']
        #program = qa['program']

        current_sample.question_id = question_id
        current_sample.video_id = video_id
        current_sample.question = question
        current_sample.start_time = start_time
        current_sample.end_time = end_time

        #current_sample.program = program
        current_sample.choices = choices
        current_sample.situations = situations

        end_frame = int(end_time * self.fps[video_id+'.mp4'])
        
        answer = None
        if not self._dataset_type == 'test':
            answer = qa['answer']
        current_sample.answer = answer


        return current_sample



    

