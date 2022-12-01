# SR questions dataset

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import reason.utils.utils as utils
import pdb

class SRDataset(Dataset):

    def __init__(self, question_h5_path, max_samples, vocab_json, test=False):
        self.max_samples = max_samples
        question_h5 = h5py.File(question_h5_path, 'r')
        self.questions = torch.LongTensor(np.asarray(question_h5['questions'], dtype=np.int64))
        self.video_idxs = np.asarray(question_h5['video_idxs'], dtype=np.int64)
        self.question_idxs = np.asarray(question_h5['orig_idxs'], dtype=np.int64)
        self.programs, self.answers = None, None
        if 'programs' in question_h5:
            self.programs = torch.LongTensor(np.asarray(question_h5['programs'], dtype=np.int64))
        if 'answers' in question_h5:
            self.answers = np.asarray(question_h5['answers'], dtype=np.int64)
        self.vocab = utils.load_vocab(vocab_json)
        self.test = test
        #pdb.set_trace()

    def __len__(self):
        if self.max_samples:
            return min(self.max_samples, len(self.questions))
        else:
            return len(self.questions)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('index %d out of range (%d)' % (idx, len(self)))
        #pdb.set_trace()
        question = self.questions[idx]
        video_idx = self.video_idxs[idx]
        question_idx = self.question_idxs[idx]
        program = -1
        answer = -1
        if self.programs is not None:
            program = self.programs[idx] 
        if self.answers is not None and len(self.answers) != 0:
            answer = self.answers[idx]
            
        if self.test:
            return question, program, answer, video_idx, question_idx
        else:
            return question, program, answer, video_idx
