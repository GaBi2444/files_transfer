"""
Take the original h5 questions as input, replace the programs by the parsed programs
for multiple choice question, use the merged vocab to re-encode the programs and questions
"""


import os
import h5py
import json
import numpy as np
from tqdm import tqdm
import reason.utils.utils as utils
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--split', required=True)
parser.add_argument('--q_type', required=True, choices=['oe', 'mc_q', 'mc_c'])
parser.add_argument('--n_prog', required=True, type=str)
args = parser.parse_args()


input_dir = '/data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/arxiv_dec_2018/models/depictr/question_parsing/data/h5'
mc_vocab_path = '/data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/arxiv_dec_2018/models/depictr/question_parsing/data/tbd/h5/mc_vocab.json'
if args.q_type == 'mc_c':
    filename = 'mc_{}_choices.h5'.format(args.split)
elif args.q_type == 'mc_q':
    filename = 'mc_{}_questions.h5'.format(args.split)
else:
    filename = 'oe_{}_questions.h5'.format(args.split)
input_h5_path = os.path.join(input_dir, filename)

output_dir = 'data/tbd/h5/{}pg'.format(args.n_prog)
output_h5_path = os.path.join(output_dir, filename)

pg_dir = '/data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/arxiv_dec_2018/models/depictr/question_parsing/data/tbd/parse_results'
if args.q_type == 'mc_c' and args.n_prog != 'all':
    n_prog = str(int(args.n_prog) * 4)
else:
    n_prog = args.n_prog
pg_files = ['{}_{}_{}pg.json'.format(args.q_type, args.split, n_prog)]
pg_paths = [os.path.join(pg_dir, f) for f in pg_files]

if args.q_type == 'oe':
    vocab_path = os.path.join(input_dir, 'oe_vocab.json')
else:
    vocab_path = mc_vocab_path
vocab = utils.load_vocab(vocab_path)

orig_vocab_path = None
if args.q_type == 'mc_q':
    orig_vocab_path = os.path.join(input_dir, 'mc_question_vocab.json')
elif args.q_type == 'mc_c':
    orig_vocab_path = os.path.join(input_dir, 'mc_choice_vocab.json')
if orig_vocab_path:
    orig_vocab = utils.load_vocab(orig_vocab_path)

def encode_program(pg, token_to_idx, length):
    output = []
    output.append(token_to_idx['<START>'])
    for m in pg:
        output.append(token_to_idx[m])
    output.append(token_to_idx['<END>'])
    while len(output) < length:
        output.append(token_to_idx['<NULL>'])
    return np.asarray(output[:length], dtype=np.int32)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

pgs = []
for p in pg_paths:
    with open(p) as f:
        pgs.append(json.load(f))

all_pgs = {}
for pg in pgs:
    all_pgs.update(pg)

with h5py.File(input_h5_path, 'r') as input_h5:
    answers = np.asarray(input_h5['answers'], dtype=np.int32)
    image_idxs = np.asarray(input_h5['video_idxs'], dtype=np.int32)
    orig_idxs = np.asarray(input_h5['orig_idxs'], dtype=np.int32)
    programs = np.asarray(input_h5['programs'], dtype=np.int32)
    question_idxs = np.asarray(input_h5['question_idxs'], dtype=np.int32)
    questions = np.asarray(input_h5['questions'], dtype=np.int32)

    N = len(input_h5['questions'])
    lp = len(input_h5['programs'][0])

# Re-encode mc questions with merged vocab
if args.q_type != 'oe':
    for i in range(len(questions)):
        for j in range(len(questions[0])):
            questions[i][j] = vocab['question_token_to_idx'][orig_vocab['question_idx_to_token'][questions[i][j]]]

if args.q_type != 'mc_c':
    for i in tqdm(range(N)):
        if args.q_type == 'oe':
            parsed_program = all_pgs[str(image_idxs[i])][str(question_idxs[i])][0]  # open ended
        else:
            parsed_program = all_pgs[str(image_idxs[i])][str(question_idxs[i])]['question_program']  # multiple choice question
        programs[i] = encode_program(parsed_program, vocab['program_token_to_idx'], lp)
else:
    i = 0
    while i < N:
        choices = all_pgs[str(image_idxs[i])][str(question_idxs[i])]['choices']
        qid = question_idxs[i]
        for c in choices:
            if question_idxs[i] != qid:
                print('Something is wrong')
                exit()
            programs[i] = encode_program(c['program'], vocab['program_token_to_idx'], lp)
            answers[i] = vocab['answer_token_to_idx'][c['answer']]
            i += 1

print(output_h5_path)
with h5py.File(output_h5_path, 'w') as output_h5:
    output_h5.create_dataset('answers', data=answers)
    output_h5.create_dataset('video_idxs', data=image_idxs)
    output_h5.create_dataset('orig_idxs', data=orig_idxs)
    output_h5.create_dataset('programs', data=programs)
    output_h5.create_dataset('question_idxs', data=question_idxs)
    output_h5.create_dataset('questions', data=questions)
