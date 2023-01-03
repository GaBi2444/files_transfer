import json
import torch
import pickle
import argparse
import copy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from IPython import embed
import numpy as np
from sklearn.metrics import average_precision_score
def load_star(star_path, qa_path, dtype='all'):
    qa_path_ = star_path + qa_path
    STAR_train, STAR_val, STAR_test = [], [], []
    if 'train' in dtype:
        print('='*10, 'Loading STAR Train Dataset', '='*10)
        STAR_train = json.load(open(qa_path_+'STAR_train.json'))

    if 'val' in dtype:
        print('='*10, 'Loading STAR Validation Dataset', '='*10)
        STAR_val = json.load(open(qa_path_+'STAR_val.json'))

    if 'test' in dtype:
        print('='*10, 'Loading STAR Test Dataset', '='*10)
        STAR_test = json.load(open(qa_path_+'STAR_test.json'))

    return STAR_train, STAR_val, STAR_test

def load_act_cls(path):
    cls_dict = {}
    for line in open(path,'r'):
        line = line.strip('/n').split()
        act_id = line[0]
        cls = ' '.join(line[1:])
        cls_dict[cls] = act_id
    return cls_dict


def load_image_lists(frame_list_file, prefix="", return_list=False):
    """
    Load image paths and labels from a "frame list".
    Each line of the frame list contains:
    `original_vido_id video_id frame_id path labels`
    Args:
        frame_list_file (string): path to the frame list.
        prefix (str): the prefix for the path.
        return_list (bool): if True, return a list. If False, return a dict.
    Returns:
        image_paths (list or dict): list of list containing path to each frame.
            If return_list is False, then return in a dict form.
        labels (list or dict): list of list containing label of each frame.
            If return_list is False, then return in a dict form.
    """
    image_paths = defaultdict(list)
    labels = defaultdict(list)
    with open(frame_list_file, "r") as f:
        assert f.readline().startswith("original_vido_id")
        for line in f:
            row = line.split()
            # original_vido_id video_id frame_id path labels
            assert len(row) == 5
            video_name = row[0]
            if prefix == "":
                path = row[3]
            else:
                path = os.path.join(prefix, row[3])
            image_paths[video_name].append(path)
            frame_labels = row[-1].replace('"', "")
            if frame_labels != "":
                labels[video_name].append(
                    [int(x) for x in frame_labels.split(",")]
                )
            else:
                labels[video_name].append([])

    if return_list:
        keys = image_paths.keys()
        image_paths = [image_paths[key] for key in keys]
        labels = [labels[key] for key in keys]
        return image_paths, labels
        
    return dict(image_paths), dict(labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate STAR csv files for SlowFast codebase")
    parser.add_argument("--star_path",default='/nobackup/users/bowu/data/STAR/',help='path to STAR dataset/annotations/frames')
    parser.add_argument("--qa_path", default='Question_Answer_SituationGraph/GT/',help='path to QA dataset')
    parser.add_argument("--anno_path", default='Annotations/',help='path to annotations')
    parser.add_argument("--csv_path", default='/nobackup/users/bowu/data/STAR/Situation_Video_Data/')
    parser.add_argument("--save_path", default='/nobackup/users/bowu/code/STAR_code/STAR_Action/exp/MViTv2_STAR_TEST_TMP/',help='path to save json')
    parser.add_argument("--charades_cls_file", default='/nobackup/users/bowu/data/Charades_v1_480/classes.txt',help='path to charades class file')
    parser.add_argument("--det_result_path", default='/nobackup/users/bowu/code/STAR_code/STAR_Action/exp/MViTv2_STAR_TEST_TMP/')
    parser.add_argument("--Charades_STAR_mapping_path", default='/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Charades_to_STAR_mapping.json')

    parser.add_argument("--dtype", default=['train'])

    args = parser.parse_args()

    STAR_train, STAR_val, STAR_test = load_star(args.star_path, args.qa_path, args.dtype)
    qa_dict = {
    'train': STAR_train
    #'val': STAR_val
    }
    Charades_STAR_mapping = json.load(open(args.Charades_STAR_mapping_path, 'r'))
    for dtype in args.dtype:
        print('='*10,dtype,'='*10)
        fps = pickle.load(open(args.star_path + args.anno_path + 'fps', 'rb'))
        det_result = pickle.load(open(args.det_result_path+'Charades_train_3_32_20.pkl','rb'))
        preds, labels = det_result[0], det_result[1]

        charades_map = load_act_cls(args.charades_cls_file)
        star_map = load_act_cls(args.star_path+args.anno_path+'action_classes.txt')
        remain_ids = []
        for key in charades_map:
            if key in star_map:
                remain_ids.append(charades_map[key])
        remain_ids = [int(remain_ids[i][1:]) for i in range(len(remain_ids))]
        
        #embed()
        image_paths, _ = load_image_lists(args.csv_path+'{}.csv'.format(dtype))

        keys = list(image_paths.keys())
        #print(keys)
        #embed()
        assert len(keys) == len(preds)
        det_act = {}
        det_act_video = {}
        count=0
        #print(len(qa_dict[dtype]))
        mAP_sf = False
        if mAP_sf:
            preds = preds.numpy()
            labels = labels.numpy()
            preds_star = copy.deepcopy(preds[:,remain_ids])
            labels = copy.deepcopy(labels[:,remain_ids])
            preds = preds_star[:, ~(np.all(labels == 0, axis=0))]
            labels = labels[:, ~(np.all(labels == 0, axis=0))]
            aps = [0]
            aps = average_precision_score(labels, preds, average=None)
            mean_ap = np.mean(aps)
            print("map :" + str(mean_ap))
        vids = []
        act_pred = []
        vid_qid_mapping = {}
        for i, qa in enumerate(tqdm(qa_dict[dtype])):
            vid = qa['video_id']
            qid = qa['question_id']
            start, end = qa['start'], qa['end']
            start_f, end_f = int(fps[vid+'.mp4']*start) + 1, int(fps[vid+'.mp4']*end) + 1 
            video_meta_star = vid + '_' + str(int(start_f)) + '_' + str(int(end_f))
            video_meta = Charades_STAR_mapping['vid_mapping']['STAR_to_Charades'][video_meta_star]

            try:
                meta_index = keys.index(video_meta)
            except:
                print(video_meta)
                count+=1
                continue
            pred_logits = copy.deepcopy(preds[meta_index][remain_ids])
            sorted_logits, sorted_indice = torch.sort(pred_logits, descending=True)
            #embed()
            #info ={}
            #info['vid'] = vid
            #info['act_pred'] = sorted_indice.tolist()
            #det_act[qid] = info
            cha_pred_logits = copy.deepcopy(preds[meta_index])
            cha_sorted_logits, cha_sorted_indice = torch.sort(cha_pred_logits, descending=True)
            #embed()
            vids.append(video_meta)
            act_pred.append(cha_sorted_indice.tolist())

            vid_qid_mapping[qid] = video_meta
            #vid_qid_mapping[video_meta_star] = qid
            det_act[qid] = sorted_indice.tolist()
        df = pd.DataFrame({
             'vid':vids,
             'act_pred':act_pred
        })
        #df.to_csv('star_test_act_pred.csv', sep = '\t', index=False)

        print('Not match num:',count)
        save_path = args.save_path + 'STAR_' + dtype + '_3_32_30.json'
        #with open(save_path,'w') as f:
        #    f.write(json.dumps(det_act))
    
        with open('../../exp/test_graph/STAR_' + dtype + '_qid_vid_mapping.json','w') as f:
            f.write(json.dumps(vid_qid_mapping))
            
