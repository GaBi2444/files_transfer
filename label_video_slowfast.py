import csv
import pandas as pd
from IPython import embed
from tqdm import tqdm
import os
import json
from collections import Counter
mode = 'val'
label_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Situation_Video_Data/" + "Charades_Cls/" + mode + ".csv"
#label_path = mode + '.csv'
gt_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Question_Answer_SituationGraph/STAR_" + mode + ".json"
STAR_test_gt_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Question_Answer_SituationGraph/STAR_test.json"
STAR_train_gt_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Question_Answer_SituationGraph/STAR_train.json"
STAR_val_gt_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Question_Answer_SituationGraph/STAR_val.json"

star_train_label_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Situation_Video_Data/Charades_Cls/train.csv"
star_test_label_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Situation_Video_Data/Charades_Cls/test.csv"
star_val_label_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Situation_Video_Data/Charades_Cls/val.csv"
charades_val_label_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/Charades/val.csv"
charades_train_label_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/Charades/train.csv"
max_frame_charade_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/Charades/charades_max_frame_ids.json"
max_frame_star_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Situation_Video_Data/max_frame_ag.json"

det_act_train_path = "../../exp/test_graph/kf_act_pred_result_train_3_32_20.json"
det_act_train = json.load(open(det_act_train_path))
det_act_val_path = "../../exp/test_graph/kf_act_pred_result_val_3_32_20.json"
det_act_val = json.load(open(det_act_val_path))
det_act_test_path = "../../exp/test_graph/kf_act_pred_result_test_3_32_10.json"
det_act_test = json.load(open(det_act_test_path))
STAR_Charades_vid_mapping_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Charades_to_STAR_mapping.json"
STAR_Charades_vid_mapping = json.load(open(STAR_Charades_vid_mapping_path))
STAR_test_qid_vid_mapping_path = "../../exp/test_graph/STAR_test_qid_vid_mapping.json"
STAR_test_qid_vid_mapping = json.load(open(STAR_test_qid_vid_mapping_path))
STAR_train_qid_vid_mapping_path = "../../exp/test_graph/STAR_train_qid_vid_mapping.json"
STAR_train_qid_vid_mapping = json.load(open(STAR_train_qid_vid_mapping_path))
STAR_val_qid_vid_mapping_path = "../../exp/test_graph/STAR_val_qid_vid_mapping.json"
STAR_val_qid_vid_mapping = json.load(open(STAR_val_qid_vid_mapping_path))
kf_mapping_test_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Situation_Video_Data/STAR_test_to_Charades_KF.json"
kf_Charades2STAR_test = json.load(open(kf_mapping_test_path))
kf_mapping_train_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Situation_Video_Data/STAR_train_to_Charades_KF.json"
kf_Charades2STAR_train = json.load(open(kf_mapping_train_path))
kf_mapping_val_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Situation_Video_Data/STAR_val_to_Charades_KF.json"
kf_Charades2STAR_val = json.load(open(kf_mapping_val_path))
#embed()
max_frame_charade = json.load(open(max_frame_charade_path))  # "xxxxxx"
max_frame_star = json.load(open(max_frame_star_path))  # "xxxxx.mp4"
STAR_train = json.load(open(STAR_train_gt_path))
STAR_test = json.load(open(STAR_test_gt_path))
STAR_val = json.load(open(STAR_val_gt_path))
# star_all_label = dict(Counter(star_train_label) + Counter(star_test_label) + Counter(star_val_label))

det_act_convert = True
pred_result = {}
STAR_qid_vid_mappings = [STAR_train_qid_vid_mapping, STAR_test_qid_vid_mapping, STAR_val_qid_vid_mapping]
det_act = [det_act_train, det_act_test, det_act_val]
kf_Charades2STAR = [kf_Charades2STAR_train, kf_Charades2STAR_test, kf_Charades2STAR_val]
mapping_mode = ['train', 'test', 'val']
if det_act_convert:
    for i,STAR_qid_vid_mapping in enumerate(STAR_qid_vid_mappings):
        for qid in STAR_qid_vid_mapping:
            vid = STAR_qid_vid_mapping[qid]
            vid_raw, _, _ = vid.split('_')

            STAR_vid = STAR_Charades_vid_mapping['vid_mapping']['Charades_to_STAR'][vid]
            fid_info = {}
            for fid in det_act[i][vid]:

                STAR_fid = kf_Charades2STAR[i][vid_raw]['Charades_To_STAR'][fid]
                fid_info[STAR_fid] = det_act[i][vid][fid]
            pred_result[qid] = fid_info
        #embed()
        compute_init_graph_act = True
        if compute_init_graph_act:
            frame_level_acc = []
            for video in STAR_test:
                situations = video['situations']
                vid = video['video_id']
                qid = video['question_id']
                if qid not in pred_result:
                    print(qid)
                    continue
                for fid in situations:
                    total_act_nums_frame = 0
                    correct_act_nums_frame = 0
                    frame_acts_gt = situations[fid]['actions']
                    frame_acts_gt = [int(x[1:]) for x in frame_acts_gt]
                    gt_act_num = len(frame_acts_gt)
                    if fid not in pred_result[qid]: continue
                    det_acts = pred_result[qid][fid]
                    total_act_nums_frame += gt_act_num
                    for act in det_acts:
                        if act in frame_acts_gt:
                            correct_act_nums_frame += 1
                    #frame_level_acc.append(correct_act_nums_frame / total_act_nums_frame)
            #frame_sum_acc = sum(frame_level_acc) / len(frame_level_acc)
            #print("frame-level acc: " + str(frame_sum_acc))
        #embed()
        with open('../../exp/test_graph/STAR_' + mapping_mode[i] + '_temporal_act.json', 'w') as f:
            f.write(json.dumps(pred_result))

keyframe_mapping = False
kf_mapping = {}
STAR_test_charades_ids = {}
STAR = [STAR_train, STAR_test, STAR_val]
dataset_mode = ['train', 'test', 'val']
if keyframe_mapping:
    for ind, dataset in enumerate(STAR):
        for q in dataset:
            vid = q['video_id']
            charades_max_frames = max_frame_charade[vid]
            star_max_frames = max_frame_star[vid + '.mp4']
            fid_mapping = {}
            star2charades = {}
            charades2star = {}
            for star_fid in q['situations']:
                charades_fid = int(int(star_fid) * charades_max_frames / star_max_frames)
                charades_fid = str(charades_fid).zfill(6)
                star2charades[star_fid] = charades_fid
                charades2star[charades_fid] = star_fid
                if vid not in STAR_test_charades_ids:
                    STAR_test_charades_ids[vid] = [charades_fid]
                else:
                    STAR_test_charades_ids[vid].append(charades_fid)
            fid_mapping['STAR_To_Charades'] = star2charades
            fid_mapping['Charades_To_STAR'] = charades2star
            if vid not in kf_mapping:
                kf_mapping[vid] = fid_mapping
            else:
                for fid in fid_mapping['Charades_To_STAR']:
                    if fid in kf_mapping[vid]['Charades_To_STAR']:
                        continue
                    else:
                        kf_mapping[vid]['Charades_To_STAR'][fid] = fid_mapping['Charades_To_STAR'][fid]
                for fid in fid_mapping['STAR_To_Charades']:
                    if fid in kf_mapping[vid]['STAR_To_Charades']:
                        continue
                    else:
                        kf_mapping[vid]['STAR_To_Charades'][fid] = fid_mapping['STAR_To_Charades'][fid]

        with open('/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Situation_Video_Data/STAR_' + dataset_mode[ind] + '_to_Charades_KF.json', 'w') as f:
            f.write(json.dumps(kf_mapping))
        with open('/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Situation_Video_Data/STAR_' + dataset_mode[ind] + '_Charades_ids.json', 'w') as f1:
            f1.write(json.dumps(STAR_test_charades_ids))

match = False
if match == True:
    label_paths = [star_train_label_path, star_test_label_path, star_val_label_path]
    star_train = {}
    star_test = {}
    star_val = {}
    STAR_raw_video = [star_train, star_test, star_val]
    for i, label_path in enumerate(label_paths):
        with open(label_path, "r") as f:
            assert f.readline().startswith("original_vido_id")

            for line in f:
                row = line.split()  # ['VZE8E_1_65', 'VZE8E', '1', 'VZE8E/VZE8E-000001.jpg', '""""']
                vid = row[0][:5]
                clip = row[0][6:]
                if vid not in STAR_raw_video[i]:
                    STAR_raw_video[i][vid] = [clip]
                else:
                    if clip in STAR_raw_video[i][vid]:
                        continue
                    else:
                        STAR_raw_video[i][vid].append(clip)

                # original_vido_id video_id frame_id path labels

    original_vido_id = []
    video_id, frame_id = [], []
    path, labels = [], []
    Charades_val = {}
    Charades_raw_video = {}

    charades_label_paths = [charades_train_label_path, charades_val_label_path]
    for charades_label_path in charades_label_paths:
        with open(charades_label_path, "r") as f:
            assert f.readline().startswith("original_vido_id")
            for line in f:
                frame_info = []
                row = line.split()  # YSKX3 7811 43 YSKX3/YSKX3-000044.jpg ""
                # original_vido_id video_id frame_id path labels
                # embed()
                # print(row)
                assert len(row) == 5

                original_vido_id.append(row[0])
                for i in range(1, 5):
                    frame_info.append(row[i])
                if row[0] not in Charades_raw_video:
                    Charades_raw_video[row[0]] = [frame_info]
                else:
                    Charades_raw_video[row[0]].append(frame_info)
    # embed()
    dtype = ['train', 'test', 'val']
    charades_to_star_vid_mapping = {}
    star2cha = {}
    cha2star = {}
    charades_to_star_vid_mapping['Charades_to_STAR'] = cha2star
    charades_to_star_vid_mapping['STAR_to_Charades'] = star2cha
    max_frame_mapping = {}
    for j, raw_video in enumerate(STAR_raw_video):
        cha_original_vido_id = []
        cha_video_id, cha_frame_id = [], []
        cha_path, cha_labels = [], []
        for vid in tqdm(raw_video, desc='Processing'):
            
            clips = raw_video[vid]
            #if vid == '02SKC': print(clips)
            charades_max_frames = max_frame_charade[vid]
            star_max_frames = max_frame_star[vid + '.mp4']
            vid_max_frame_mapping = {}
            vid_max_frame_mapping['Charades'] = charades_max_frames
            vid_max_frame_mapping['STAR'] = star_max_frames
            if vid not in Charades_raw_video: continue
            max_frame_mapping[vid] = vid_max_frame_mapping
            for clip in clips:
                star_start, star_end = clip.split('_')
                charades_start = int(int(star_start) * charades_max_frames / star_max_frames) - 1
                charades_start = max(charades_start, 1)
                charades_end = int(int(star_end) * charades_max_frames / star_max_frames) + 1
                charades_end = min(charades_end, charades_max_frames)
                clip_start_end_dense = list(range(charades_start, charades_end))
                star_raw_id = vid + '_' + str(star_start) + '_' + str(star_end)
                charades_raw_id = vid + '_' + str(charades_start) + '_' + str(charades_end)

                cha2star[charades_raw_id] = star_raw_id
                star2cha[star_raw_id] = charades_raw_id
                for i in clip_start_end_dense:
                    cha_original_vido_id.append(charades_raw_id)
                    cha_video_id.append(vid)
                    cha_frame_id.append(i)
                    cha_path.append(Charades_raw_video[vid][i - 1][2])
                    cha_labels.append(Charades_raw_video[vid][i - 1][3])
        df = pd.DataFrame({
            'original_vido_id': cha_original_vido_id,
            'video_id': cha_video_id,
            'frame_id': cha_frame_id,
            'path': cha_path,
            'labels': cha_labels
        })
        df.to_csv("/gpfs/u/home/NSVR/NSVRbowu/scratch/data/Charades/STAR_split/" + dtype[j] + ".csv", sep='\t', index = False)
    Charades_to_STAR_mapping = {}
    charades_to_star_vid_mapping['Charades_to_STAR'] = cha2star
    charades_to_star_vid_mapping['STAR_to_Charades'] = star2cha
    Charades_to_STAR_mapping['vid_mapping'] = charades_to_star_vid_mapping
    Charades_to_STAR_mapping['max_frame_mapping'] = max_frame_mapping
    with open("/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Charades_to_STAR_mapping.json", "w") as f:
        f.write(json.dumps(Charades_to_STAR_mapping))
    print('finish')
    
only_convert = False
if only_convert == True:
    gt_label = json.load(open(gt_path))
    original_vido_id = []
    video_id, frame_id = [], []
    path, labels = [], []
    gt_label_query = {}
    STAR_raw_video = []
    vids_index = {}
    for qid in gt_label:
        vid = qid['video_id']
        situations = qid['situations']
        video_label = {}
        for fid in situations:
            acts_label = situations[fid]['actions']
            if len(acts_label) == 0: continue
            acts_label = [str(int(x[1:])) for x in acts_label]
            if fid not in video_label:
                video_label[fid] = acts_label
        if len(video_label) == 0: continue
        if vid not in gt_label_query:
            gt_label_query[vid] = video_label
    with open(label_path, "r") as f:
        assert f.readline().startswith("original_vido_id")
        for line in f:
            row = line.split()  # ['VZE8E_1_65', 'VZE8E', '1', 'VZE8E/VZE8E-000001.jpg', '""""']
            STAR_raw_video.append(row[0])
            # original_vido_id video_id frame_id path labels
            # embed()
            # print(row)
            assert len(row) == 5
            vid = row[1]
            fid = row[3][-10:-4]
            aimos_path = vid + '.mp4' + '/' + fid + '.png'
            # img_path = '/nobackup/users/bowu/data/ActionGenome/dataset/ag/frames/' + aimos_path
            # if not os.path.exists(img_path):
            # print("not exist: " + str(img_path))
            #    continue
            vids_index[vid] = 1
            # if len(vids_index) % 100 == 0 : print(len(vids_index))
            original_vido_id.append(row[0])
            video_id.append(row[1])
            frame_id.append(row[2])

            # path.append(ro
            label_ = '""'
            path.append(aimos_path)
            if vid in gt_label_query:
                if fid in gt_label_query[vid]:
                    # if vid =='02SKC' and fid =='000593':
                    #    embed()
                    frame_labels = gt_label_query[vid][fid]
                    label_ = '"' + ','.join(frame_labels) + '"'
            else:
                label_ = '""'

            # labels.append(label_)
            labels.append(row[4])
    df = pd.DataFrame({
        'original_vido_id': original_vido_id,
        'video_id': video_id,
        'frame_id': frame_id,
        'path': path,
        'labels': labels
    })
    df.to_csv("/gpfs/u/home/NSVR/NSVRbowu/scratch/data/Charades/STAR_split" + mode + ".csv", sep='\t', index = False)


