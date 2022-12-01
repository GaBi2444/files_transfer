"""
Merge the parsed multiple choice questions and choices
"""
import os
import json
import argparse
import h5py
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--question_path', required=True)
parser.add_argument('--choice_path', required=True)
parser.add_argument('--output_path', required=True)
parser.add_argument('--raw_question_path', required=True)
args = parser.parse_args()


with open(args.question_path) as f:
    mc_q_pg = json.load(f)
with open(args.choice_path) as f:
    mc_c_pg = json.load(f)

with open(args.raw_question_path, 'r') as f:
    raw_ques_list = json.load(f)

new_ques_dict = {}
for i, s in mc_q_pg.items():
    tmp_ques_info = {}
    int_i = int(i)
    raw_ques_info = raw_ques_list[int_i]
    tmp_ques_info['scene_index'] = raw_ques_info['scene_index']
    tmp_ques_info['video_filename'] = raw_ques_info['video_filename']
    ques_list = [[] for ii in range(len(s))]
    for j, q in s.items():
        int_j = int(j)
        q['question'] = raw_ques_info['questions'][int_j]['question']
        q['question_type'] = raw_ques_info['questions'][int_j]['question_type'] 
        q['question_subtype'] = raw_ques_info['questions'][int_j]['program'][-1] 
        q['program_gt'] = raw_ques_info['questions'][int_j]['program']
        q['choices'] = mc_c_pg[i][j]['choices']
        pdb.set_trace()
        for c_id, choice in enumerate(q['choices']):
            if choice['answer']=='yes':
                choice['answer'] = 'correct'
            if choice['answer']=='no':
                choice['answer'] = 'wrong'
        ques_list[int_j] = q
    tmp_ques_info['questions'] = ques_list
    new_ques_dict[i] = tmp_ques_info 

with open(args.output_path, 'w') as fout:
    json.dump(new_ques_dict, fout)
