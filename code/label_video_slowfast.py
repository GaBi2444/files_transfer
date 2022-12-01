import csv
import pandas as pd
from IPython import embed
from tqdm import tqdm
import os
import json
from collections import Counter
mode = 'train'
label_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Situation_Video_Data/" + "Charades_Cls/" + mode + ".csv"
#label_path = mode + '.csv'
gt_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Question_Answer_SituationGraph/STAR_" + mode + ".json"

star_train_label_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Situation_Video_Data/Charades_Cls/train.csv"
star_test_label_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Situation_Video_Data/Charades_Cls/test.csv"
star_val_label_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Situation_Video_Data/Charades_Cls/val.csv"
charades_label_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/Charades/val.csv"
max_frame_charade_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/Charades/charades_max_frame_ids.json"
max_frame_star_path = "/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR/Situation_Video_Data/max_frame_ag.json"
max_frame_charade = json.load(open(max_frame_charade_path)) #"xxxxxx"
max_frame_star = json.load(open(max_frame_star_path)) #"xxxxx.mp4"
#star_all_label = dict(Counter(star_train_label) + Counter(star_test_label) + Counter(star_val_label))
STAR_raw_video = {}
match = True
if match == True:
    label_paths = [star_train_label_path, star_test_label_path, star_val_label_path]
    for label_path in label_paths:
        with open(label_path, "r") as f:
            assert f.readline().startswith("original_vido_id")

            for line in f:
                row = line.split() #['VZE8E_1_65', 'VZE8E', '1', 'VZE8E/VZE8E-000001.jpg', '""""']
                vid = row[0][:5]
                clip = row[0][6:]
                if vid not in STAR_raw_video:
                    STAR_raw_video[vid] = [clip]
                else:
                    if clip in STAR_raw_video[vid]: continue
                    else:
                        STAR_raw_video[vid].append(clip)

                # original_vido_id video_id frame_id path labels

    original_vido_id = []
    video_id, frame_id = [], []
    path, labels = [], []
    Charades_val = {}
    with open(charades_label_path, "r") as f:
        assert f.readline().startswith("original_vido_id")
        total_lines = len(f.readlines())
        f.close()
    frame_info = []
    Charades_video = {}
    with open(charades_label_path, "r") as f:
        assert f.readline().startswith("original_vido_id")

        for line in f:
            row = line.split() #YSKX3 7811 43 YSKX3/YSKX3-000044.jpg ""
            # original_vido_id video_id frame_id path labels
            #embed()
            #print(row)
            assert len(row) == 5
            
            original_vido_id.append(row[0])
            for i in range(1,4):
                frame_info.append(row[i])
            if row[0] not in Charades_video:
                Charades_video[row[0]] = [frame_info]
            else:
                Charades_video[row[0]].append(frame_info)
            
    
    cha_original_vido_id = []
    cha_video_id, cha_frame_id = [], []
    cha_path, cha_labels = [], []
    for vid in tqdm(STAR_raw_video, desc='Processing'):
        clips = STAR_raw_video[vid]
        charades_max_frames = max_frame_charade[vid]
        star_max_frames = max_frame_star[vid + '.mp4']
        for clip in clips:
            star_start, star_end = clip.split('_')
            charades_start = int(int(star_start) * charades_max_frames / star_max_frames)
            charades_end = int(int(star_end) * charades_max_frames / star_max_frames)
            clip_start_end_dense = list(range(charades_start, charades_end + 1))
            for i in clip_start_end_dense:
                cha_original_vido_id.append(vid + '_' + str(charades_start) + '_' + str(charades_end))
                cha_video_id.append(vid)
                cha_frame_id.append(i)
                cha_path.append(Charades_video[vid][i-1][2])
                cha_labels.append(Charades_video[vid][i-1][3])
    df = pd.DataFrame({
            'original_video_id':cha_original_vido_id,
            'video_id':cha_video_id,
            'frame_id':cha_frame_id,
            'path':cha_path,
            'labels':cha_labels
        })
    df.to_csv("/gpfs/u/home/NSVR/NSVRbowu/scratch/data/Charades/STAR_split/" + mode + ".csv", sep='\t', index = False)
                

    

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
        if len(video_label)==0:continue
        if vid not in gt_label_query:
            gt_label_query[vid] = video_label
    with open(label_path, "r") as f:
        assert f.readline().startswith("original_vido_id")
        for line in f:
            row = line.split() #['VZE8E_1_65', 'VZE8E', '1', 'VZE8E/VZE8E-000001.jpg', '""""']
            STAR_raw_video.append(row[0])
            # original_vido_id video_id frame_id path labels
            #embed()
            #print(row)
            assert len(row) == 5
            vid = row[1]
            fid = row[3][-10:-4]
            aimos_path = vid + '.mp4' + '/' + fid + '.png'
            #img_path = '/gpfs/u/home/NSVR/NSVRbowu/scratch/data/ActionGenome/dataset/ag/frames/' + aimos_path
            #if not os.path.exists(img_path):
                #print("not exist: " + str(img_path))
            #    continue
            vids_index[vid] = 1
            #if len(vids_index) % 100 == 0 : print(len(vids_index))
            original_vido_id.append(row[0])
            video_id.append(row[1])
            frame_id.append(row[2])

            #path.append(ro
            label_ = '""'
            path.append(aimos_path)
            if vid in gt_label_query:
                if fid in gt_label_query[vid]:
                    #if vid =='02SKC' and fid =='000593':
                    #    embed()
                    frame_labels = gt_label_query[vid][fid]
                    label_ = '"' +  ','.join(frame_labels) + '"'
            else:
                label_ = '""
                                                                   
             #labels.append(label_)
            labels.append(row[4])
    df = pd.DataFrame({
            'original_vido_id':original_vido_id,
            'video_id':video_id,
            'frame_id':frame_id,
            'path':path,
            'labels':labels
        })
    df.to_csv("/gpfs/u/home/NSVR/NSVRbowu/scratch/data/Charades/STAR_split" + mode + ".csv", sep='\t', index = False)
