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
from mmf.datasets.builders.star.dataset import STARDataset
from mmf.utils.text import VocabFromText, tokenize
from mmf.utils.general import get_mmf_root
from PIL import Image

from mmf.datasets.builders.star.utils import *

logger = logging.getLogger(__name__)

_QTYPE = ['Interaction', 'Sequence', 'Prediction', 'Feasibility']

_CONSTANTS = {
    "OBJ_NUM": 37, 
    "ACT_NUM": 111,
    "REL_NUM": 24,
    "dataset_key": 'star_nssr'
}

_TEMPLATES = {
    "data_folder_missing_error": "Data folder {} for STAR is not present.",
}

class STARDataset_NSSR(STARDataset):

    def __init__(self, config, dataset_type, data_folder=None, *args, **kwargs):
        super().__init__(config, dataset_type,dataset_name="star_nssr", data_folder=None, *args, **kwargs)

        self.fps = pickle.load(open(os.path.join(self._data_folder,self.anno_folder,'fps'),'rb'))

        self.max_act = config.max_act
        self.max_frame = config.max_frame
        self.max_rel = config.max_rel

        self.visual_feature_type = config.visual_feature_type
        self.fea_h, self.fea_w  = config.fea_h, config.fea_w

        # feature setting
        self.symbolic = config.symbolic
        self.semantic = config.semantic
        self.visual = config.visual

        # using detected action
        self.use_act_det = False
        if config.act_det != '':
            self.use_act_det = True
            self.det_act = json.load(open(os.path.join(config.data_dir, config.feature_folder, 'ActRecog/', config.act_det, 'STAR_{}.json'.format(self._dataset_type))))

        self.visual_feature_path = os.path.join(self._data_dir, config.feature_folder, 'video_features', config.visual_feature_type)
        self.hyper_token = init_hyper_token(self.max_act, self.max_rel, self.max_frame)
        self.type_token = init_type_token(self.max_act, self.max_rel, self.max_frame)
        self.triplet_token = init_triplet_token(self.max_act, self.max_rel, self.max_frame)
        self.situation_token = init_situation_token(self.max_act, self.max_rel, self.max_frame)
        self.special_id, self.special_target = init_special_token(self.max_rel,self.max_frame)
        self.con_act = config.con_act
        self.graph_type = config.graph_type

        self.vocab_file = os.path.join(self._data_dir, config.data_folder, 'Vocab','star_vocab.txt')

        self.init_cls_mapping()
        self.build_vocab()

    def load_json(self):
        if self._dataset_type=='train':
            graph = self.train_graph
        else:
            graph = self.val_graph

        if graph=='GT':
            with open(os.path.join(self._data_folder, self.qa_folder, graph, self.qa_json)) as f:
                self.sr_qa = json.load(f)
        else:
            with open(os.path.join(self._data_folder, self.qa_folder, graph, self.graph_type, self.qa_json)) as f:
                self.sr_qa = json.load(f)

    def build_vocab(self):

        if os.path.exists(self.vocab_file):
            return

        os.makedirs(os.path.dirname(self.vocab_file), exist_ok=True)

        word_list = []

        key = sorted(self.id2word.keys())

        for k in key:
            for w in self.id2word[k]:
                if w not in word_list:
                    word_list.append(w)

        with open(self.vocab_file, "w") as f:
            f.write("\n".join(word_list))

    def load_visual_feature(self, select_keyframe, vid):

        features = []
        for f in select_keyframe:
            if 'padding' in f:
                features.append(torch.zeros([512,14,14]))
            else:
                feature = np.load(os.path.join(self.visual_feature_path , vid , f + '.npy'))
                feature = torch.from_numpy(feature)[:,4,:,:]
                features.append(feature)
                #features.append(torch.randn([512,14,14]))

        return torch.stack(features)

    def filter_word(self, word_list, remove=['a', 'the', 'some', 'somewhere', 's']):
        filtered = []
        for word in word_list:
            if word in remove:
                continue
            if '/' in word:
                word = word.split('/')
                for w in word:
                    if w in remove:
                        continue
                    else:
                        filtered.append(w)
            else:
                filtered.append(word)

        return filtered

    def init_cls_mapping(self):
        self.id2cls = {}
        self.id2word ={}
        self.id2cls[0] = '<pad>'
        self.id2cls[1] = '<unk>'
        self.id2cls[2] = '<sep>'
        self.id2cls[3] = '<mask>'

        self.id2word[0] = ['<pad>']
        self.id2word[1] = ['<unk>']
        self.id2word[2] = ['<sep>']
        self.id2word[3] = ['<mask>']
        index = 4

        with open(os.path.join(self._data_folder, self.anno_folder, "action_classes.txt")) as f:
            lines = f.readlines()
            for line in lines:
                mapping = line.split()
                self.id2cls[index] = mapping[0]
                filtered_words = self.filter_word(mapping[1:])
                self.id2word[index] = filtered_words
                index+=1

        with open(os.path.join(self._data_folder, self.anno_folder, "object_classes.txt")) as f:
            lines = f.readlines()
            for line in lines:
                mapping = line.split()
                self.id2cls[index] = mapping[0]
                words = mapping[1].split('/')
                filtered_words = self.filter_word(words)
                self.id2word[index] = filtered_words
                index += 1

        with open(os.path.join(self._data_folder, self.anno_folder, "relationship_classes.txt")) as f:
            lines = f.readlines()
            for line in lines:
                mapping = line.split()
                self.id2cls[index] = mapping[0]
                words = mapping[1].split('_')
                filtered_words = self.filter_word(words)
                self.id2word[index] = filtered_words
                index += 1

    def format_for_prediction(self,report):

        pre_act = report.pre_act_cls.tolist()
        pre_obj1 = report.pre_obj1_cls.tolist()
        pre_rel = report.pre_rel_cls.tolist()
        pre_obj2 = report.pre_obj2_cls.tolist()
        question_id = report.question_id
        select_keyframe = report.select_keyframe

        output = []
        for i , (qid, keyframes, acts, objs1, rels, objs2) in enumerate(zip(question_id,select_keyframe,pre_act,pre_obj1,pre_rel,pre_obj2)):
            situations = {}
            for kf, act, obj1, rel, obj2 in zip(keyframes, acts, objs1, rels, objs2):
                actions = [ self.id2cls[a] for a in act]
                rel_pairs = [[self.id2cls[o1],self.id2cls[o2]] for o1, o2 in zip(obj1,obj2)]
                rel_labels = [self.id2cls[r] for r in rel]
                bbox_labels = list(set([self.id2cls[o1] for o1 in obj1] + [self.id2cls[o2] for o2 in obj2]))
                situations[kf] = {}
                situations[kf]['actions'] = actions
                situations[kf]['rel_pairs'] = rel_pairs
                situations[kf]['rel_labels'] = rel_labels
                situations[kf]['bbox_labels'] = bbox_labels
            output.append([qid,situations])

        return output

    def cls2word(self, token_ids):
        input_words = []

        for i in range(len(token_ids)):
            temp = []
            for id_ in token_ids[i]:
                a = id_.tolist()
                temp.append(self.id2word[a])
            input_words.append(temp)

        return input_words


    def __len__(self):
        return len(self.sr_qa) # situation reasoning QA

    def __getitem__(self, idx):

        qa = self.sr_qa[idx]
        
        current_sample = Sample()
        question = qa["question"]
        choices = qa['choices']
        video_id, question_id = qa['video_id'], qa['question_id']
        start_time, end_time = qa['start'], qa['end']
        situations = qa['situations']
        #program = qa['question_program']

        current_sample.question_id = question_id
        current_sample.video_id = video_id
        current_sample.question = question
        current_sample.start_time = start_time
        current_sample.end_time = end_time

        #current_sample.program = program
        current_sample.choices = choices
        #current_sample.situations = situations

        answer = None
        if not self.dataset_type == 'test':
            answer = qa['answer']
        current_sample.answer = answer

        end_frame = int(end_time * self.fps[video_id+'.mp4'])

        keyframe_ids = sorted(situations.keys())
        #print('keyframe_ids', keyframe_ids)
        act, obj1, rel, obj2 = {}, {}, {}, {}
        obj_bbox = {}

        frame_path = os.path.join(self._data_folder, self.frame_folder, video_id + '.mp4/', keyframe_ids[0] + '.png')

        ref_img = Image.open(frame_path)
        w,h = ref_img.size

        for f in situations:
            act[f] = situations[f]['actions']
            rel[f] = situations[f]['rel_labels']
            obj_label = situations[f]['bbox_labels']
            resized_boxes = {}
            for i, box in enumerate(situations[f]['bbox']):
                x1, y1, x2, y2 = box[0]/w * self.fea_w, box[1]/h * self.fea_h, box[2]/w * self.fea_w, box[3]/h * self.fea_h
                resized_boxes[obj_label[i]] = [x1, y1, x2, y2]
            obj_bbox[f] = resized_boxes
            obj1[f] = [pair[0] for pair in situations[f]['rel_pairs']]
            obj2[f] = [pair[1] for pair in situations[f]['rel_pairs']]

        #print('keyframe_ids',keyframe_ids)
        select_keyframe, frame_attmask = self.graph_sampler(keyframe_ids, end_frame)
        current_sample.select_keyframe = list(sorted(select_keyframe.keys()))

        if self._dataset_type == 'train':
            act_id, act_attmask, act_target, act_label, _ = self.masked_act_processor_train(act, select_keyframe)
            rel_id, rel_attmask, rel_target, rel_label, _ = self.masked_rel_processor_train(rel, select_keyframe)
            obj1_id, obj1_attmask, obj1_target, obj1_label, _ = self.masked_obj_processor_train(obj1, select_keyframe)
            obj2_id, obj2_attmask, obj2_target, obj2_label, _ = self.masked_obj_processor_train(obj2, select_keyframe)
        else:
            act_id, act_attmask, act_target, act_label, _ = self.masked_act_processor_val(act, select_keyframe)
            rel_id, rel_attmask, _, rel_label, rel_target= self.masked_rel_processor_val(rel, select_keyframe)
            obj1_id, obj1_attmask, _, obj1_label, obj1_target= self.masked_obj_processor_val(obj1, select_keyframe)
            obj2_id, obj2_attmask, _, obj2_label, obj2_target= self.masked_obj_processor_val(obj2, select_keyframe)

        frame_feature = self.load_visual_feature(select_keyframe, video_id)
        current_sample.frame_feature = frame_feature
        current_sample.frame_attmask = frame_attmask

        current_sample.act_target = act_target
        current_sample.act_attmask = act_attmask

        current_sample.rel_target = rel_target
        current_sample.rel_attmask = rel_attmask

        current_sample.obj1_target = obj1_target
        current_sample.obj1_attmask = obj1_attmask
        
        current_sample.obj2_target = obj2_target
        current_sample.obj2_attmask = obj2_attmask

        current_sample.special_id = copy.deepcopy(self.special_id)
        current_sample.special_target = copy.deepcopy(self.special_target)
        current_sample.special_attmask = copy.deepcopy(rel_attmask)

        if self.use_act_det:
            if question_id in self.det_act:
                det_act_id = torch.tensor(self.det_act[question_id][:self.max_act])
                det_act_id = [det_act_id for i in range(self.max_frame)]
                act_id = torch.stack(det_act_id)

        if self.con_act and self._dataset_type=='test':
            
            if 'after' in question:
                condition_action = copy.deepcopy(act_target[0])
                index = condition_action==-1
                condition_action[index]=0
                act_id[0] = condition_action
            elif 'before' in question:
                condition_action = copy.deepcopy(act_target[-1])
                index = condition_action==-1
                condition_action[index]=0
                act_id[-1] = condition_action

        if self.symbolic:
            current_sample.act_id = act_id
            current_sample.obj1_id = obj1_id
            current_sample.rel_id = rel_id
            current_sample.obj2_id = obj2_id
        else:
            current_sample.act_id = None
            current_sample.obj1_id = None
            current_sample.rel_id = None
            current_sample.obj2_id = None
            
        if self.semantic:
            act_words = self.cls2word(act_id)
            obj1_words = self.cls2word(obj1_id)
            rel_words = self.cls2word(rel_id)
            obj2_words = self.cls2word(obj2_id)
            spe_words = self.cls2word(copy.deepcopy(self.special_id))

            act_semantic = self.text_processor(act_words)
            obj1_senmatic = self.text_processor(obj1_words)
            rel_semantic = self.text_processor(rel_words)
            obj2_senmatic = self.text_processor(obj2_words)
            special_semantic = self.text_processor(spe_words)

            #print('act_semantic',act_semantic.shape)
            #print('special_semantic',special_semantic.shape)

            current_sample.special_semantic = special_semantic
            current_sample.act_semantic = act_semantic
            current_sample.obj1_senmatic = obj1_senmatic
            current_sample.rel_semantic = rel_semantic
            current_sample.obj2_senmatic = obj2_senmatic
        else:
            current_sample.special_semantic = None
            current_sample.act_semantic = None
            current_sample.obj1_senmatic = None
            current_sample.rel_semantic = None
            current_sample.obj2_senmatic = None

        if self.visual:
            obj1_feature = self.visual_feature_cropper(obj1_label, obj_bbox, select_keyframe, copy.deepcopy(frame_feature))
            obj2_feature = self.visual_feature_cropper(obj2_label, obj_bbox, select_keyframe, copy.deepcopy(frame_feature))
            current_sample.obj1_feature = obj1_feature # [t, 8 ,2048]
            current_sample.obj2_feature = obj2_feature
        else:
            current_sample.obj1_feature = None # [t, 8 ,2048]
            current_sample.obj2_feature = None
            

        current_sample.type_token = torch.tensor(copy.deepcopy(self.type_token),dtype=torch.long)
        current_sample.triplet_token = torch.tensor(copy.deepcopy(self.triplet_token),dtype=torch.long)
        current_sample.hyper_token = torch.tensor(copy.deepcopy(self.hyper_token),dtype=torch.long)
        current_sample.situation_token = torch.tensor(copy.deepcopy(self.situation_token),dtype=torch.long)

        return current_sample
