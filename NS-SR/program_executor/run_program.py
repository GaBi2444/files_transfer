"""
Run symbolic reasoning on multiple-choice questions
"""
import os
import json
import argparse
import copy
from tqdm import tqdm
from executor import Executor
import numpy as np
np.random.seed(626)


def merge_situations(gt_situations,predict_situations,start,end,video_id,task):

    if task == 'action':
        merged_situations = copy.deepcopy(gt_situations)
        for frame in merged_situations:
            if frame in predict_situations:
                merged_situations[frame]['actions']= predict_situations[frame]['actions']
            else:
                merged_situations[frame]['actions']=[]
        
        return merged_situations

    if task == 'predict':
        merged_situations = {}
        original = copy.deepcopy(gt_situations)
        end_frame = int(end * fps[video_id+'.mp4'])

        for frame in original:

            if int(frame)  <= end_frame:
                if frame in predict_situations:
                    merged_situations[frame] = original[frame]
                    merged_situations[frame]['actions'] = predict_situations[frame]['actions']
                else:
                    merged_situations[frame] = original[frame]
                    merged_situations[frame]['actions'] = []
            else:
                if frame in predict_situations:
                    merged_situations[frame] = predict_situations[frame]

        for frame in predict_situations:
            if 'predict' in frame:
                frame_id = str(MAX + int(frame.split('_')[1]))
                merged_situations[frame_id] = predict_situations[frame]

        return merged_situations

def format_to_dict(situations):
    situa_dict = {}
    for situ in situations:
        situa_dict[situ[0]] = situ[1]
    return situa_dict

def Situation_Reasoning(questions, predict_situations, random_choice, label_dir, debug=False):
    
    correct = 0
    pbar = tqdm(range(len(questions)))
    predict_situations = format_to_dict(predict_situations)
    flag = ['Wrong','Correct']

    for i in pbar:
        qa = questions[i]
        question_id = qa['question_id']
        qtype, temp = question_id.split('_')[0], question_id.split('_')[1]
        qa_start = qa['start']
        qa_end = qa['end']
        video_id = qa['video_id']
        pre_situ = predict_situations[question_id]
        #compos_situation = merge_situations(qa['situations'], predict_situations[question_id],qa_start,qa_end,video_id,task)
        exe = Executor(pre_situ, label_dir)    
        q_gt_pg = qa['program']
        count = 0
        failed = 0
        need_random = True

        for choice in qa['choices']:

            c_gt_pg =  choice['op_pro']
            full_pg = q_gt_pg + c_gt_pg
            try:
                pred = exe.run(full_pg, debug=debug)
            except:
                continue

            if pred == flag[int(choice['choice']==qa['answer'])]:
                count += 1
            if pred == 'Correct':
                need_random = False

        if count == len(qa['choices']):
            correct+=1   

        # apply random choice strategy if the predict answer is ou of option
        if random_choice:
            if need_random:
                rand_choice = np.random.randint(0,4)
                if qa['choices'][rand_choice]['choice'] == qa['answer']:
                    correct+=1

        pbar.set_description('{} Question Acc {:f}'.format(qtype,float(correct)*100/len(questions)))

    return float(correct)*100/len(questions)


def multi_type_multi_times_test(args):
    qa_dir = args.qa_dir
    predict_situations_path = args.predict_situations_path
    qtypes = args.qtypes.split('_')
    rand_choice = args.rand_choice
    label_dir = args.label_dir
    test_types = []

    for q in qtypes:
        if q == 'all':
            test_types = ['Interaction','Sequence','Prediction','Feasibility']
            break
        if q == 'i':
            test_types.append('Interaction')
        if q == 's':
            test_types.append('Sequence')
        if q == 'p':
            test_types.append('Prediction')
        if q == 'f':
            test_types.append('Feasibility')

    print('----------Situation Reasoning----------')
    for qtype in test_types:

        questions = json.load(open(qa_dir + '/' + qtype + '_test.json'))
        file = 'star_'+ qtype +'_action_transition_model.json'
        predict_situations = json.load(open(predict_situations_path + file))
        
        if rand_choice:
            total_acc = 0
            for i in range(args.test_iter):
                acc = Situation_Reasoning(questions, predict_situations, args.rand_choice, label_dir)
                total_acc += acc
            print(qtype,total_acc/args.test_iter)
        else:
            total_acc = Situation_Reasoning(questions, predict_situations, args.rand_choice, label_dir)
            print(qtype,total_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Situation Reasoning")
    parser.add_argument("--qtypes",default='all')
    parser.add_argument("--qa_dir", default="/nobackup/users/bowu/data/STAR/Question_Answer_SituationGraph/GT/")
    parser.add_argument("--label_dir", default="/nobackup/users/bowu/data/STAR/Annotations/")
    parser.add_argument("--predict_situations_path",default='/Users/yushoubin/Desktop/')
    parser.add_argument("--rand_choice",default=False,type=bool)
    parser.add_argument("--test_iter",default=10,type=int)

    args = parser.parse_args()
    multi_type_multi_times_test(args)

