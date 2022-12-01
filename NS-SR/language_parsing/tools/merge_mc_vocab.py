"""
Merge the vocabularies for multiple choice questions and choices
need to re-process the h5 question files to use
"""

import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--q_vocab_path', default='data/h5/mc_question_vocab.json')
parser.add_argument('--c_vocab_path', default='data/h5/mc_choice_vocab.json')
parser.add_argument('--output_path', default='_scratch/mc_vocab.json')
args = parser.parse_args()


if not os.path.isdir(os.path.dirname(args.output_path)):
    os.makedirs(os.path.dirname(args.output_path))

with open(args.q_vocab_path) as f:
    q_vocab = json.load(f)
with open(args.c_vocab_path) as f:
    c_vocab = json.load(f)

for k in q_vocab.keys():
    for t in c_vocab[k]:
        if t not in q_vocab[k]:
            q_vocab[k][t] = len(q_vocab[k])

with open(args.output_path, 'w') as fout:
    json.dump(q_vocab, fout)