"""
Merge the parsed multiple choice questions and choices
"""
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--question_path', required=True)
parser.add_argument('--choice_path', required=True)
parser.add_argument('--output_path', required=True)
args = parser.parse_args()


with open(args.question_path) as f:
    mc_q_pg = json.load(f)
with open(args.choice_path) as f:
    mc_c_pg = json.load(f)
import pdb
pdb.set_trace()
for i, s in mc_q_pg.items():
    for j, q in s.items():
        q['choices'] = mc_c_pg[i][j]['choices']

with open(args.output_path, 'w') as fout:
    json.dump(mc_q_pg, fout)
