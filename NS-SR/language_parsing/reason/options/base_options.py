import os
import argparse

import numpy as np
import torch


class BaseOptions():
    """Base option class"""

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--run_dir', default='_scratch/test_run', type=str, help='experiment directory')
        self.parser.add_argument('--dataset', default='oe', type=str, help='select dataset, options: oe, mc_q, mc_c')
        # Dataloader
        self.parser.add_argument('--shuffle', default=1, type=int, help='shuffle dataset')
        self.parser.add_argument('--num_workers', default=1, type=int, help='number of workers for loading data')
        # Run
        self.parser.add_argument('--manual_seed', default=None, type=int, help='manual seed')
        self.parser.add_argument('--gpu_ids', default='0', type=str, help='ids of gpu to be used')
        self.parser.add_argument('--visualize', default=0, type=int, help='visualize experiment')
        # Dataset catalog
        # - Open ended
        self.parser.add_argument('--oe_train_question_path', type=str, default='data/h5/oe_train_questions.h5')
        self.parser.add_argument('--oe_val_question_path', type=str, default='data/h5/oe_val_questions.h5')
        self.parser.add_argument('--oe_test_question_path', type=str, default='data/h5/oe_test_questions.h5')
        self.parser.add_argument('--oe_vocab_path', type=str, default='data/h5/oe_vocab.json')
        # - Multiple Choice - Questions
        self.parser.add_argument('--mc_q_train_question_path', type=str, default='data/h5/mc_train_questions.h5')
        self.parser.add_argument('--mc_q_val_question_path', type=str, default='data/h5/mc_val_questions.h5')
        self.parser.add_argument('--mc_q_test_question_path', type=str, default='data/h5/mc_test_questions.h5')
        self.parser.add_argument('--mc_q_vocab_path', type=str, default='data/h5/mc_question_vocab.json')
        # - Multiple Choice - Choices
        self.parser.add_argument('--mc_c_train_question_path', type=str, default='data/h5/mc_train_choices.h5')
        self.parser.add_argument('--mc_c_val_question_path', type=str, default='data/h5/mc_val_choices.h5')
        self.parser.add_argument('--mc_c_test_question_path', type=str, default='data/h5/mc_test_choices.h5')
        self.parser.add_argument('--mc_c_vocab_path', type=str, default='data/h5/mc_choice_vocab.json')
        
    def parse(self):
        # Instantiate option
        self.opt = self.parser.parse_args()

        # Parse gpu id list
        str_gpu_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_gpu_ids:
            if str_id.isdigit() and int(str_id) >= 0:
                self.opt.gpu_ids.append(int(str_id))
        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu_ids[0])
        else:
            print('| using cpu')
            self.opt.gpu_ids = []

        # Set manual seed
        if self.opt.manual_seed is not None:
            torch.manual_seed(self.opt.manual_seed)
            if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.cuda.manual_seed(self.opt.manual_seed)

        # Print and save options
        args = vars(self.opt)
        print('| options')
        print('parameter setting')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        if not os.path.isdir(self.opt.run_dir):
            os.makedirs(self.opt.run_dir)
        file_path = os.path.join(self.opt.run_dir, 'opt.txt')
        with open(file_path, 'wt') as fout:
            fout.write('| options\n')
            for k, v in args.items():
                fout.write('%s: %s\n' % (str(k), str(v)))

        return self.opt
