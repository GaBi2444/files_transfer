import json
import csv
import pandas as pd
from IPython import embed
mode = 'val'
label_path = "wrong_action_labels/" + mode + ".csv"
#label_path = "test.csv"
gt_path = "../Question_Answer_SituationGraph/GT/STAR_" + mode + ".json"
gt_label = json.load(open(gt_path))
original_vido_id = []
video_id, frame_id = [], []
path, labels = [], []
gt_label_query = {}

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
        # original_vido_id video_id frame_id path labels
        #embed()
        #print(row)
        assert len(row) == 5
        original_vido_id.append(row[0])
        video_id.append(row[1])
        frame_id.append(row[2])
       
        #path.append(row[3])
        vid = row[1]
        fid = row[3][-10:-4]
        aimos_path = vid + '.mp4' + '/' + fid + '.png'
        label_ = '""'
        path.append(aimos_path)
        if vid in gt_label_query:
            if fid in gt_label_query[vid]:
                #if vid =='02SKC' and fid =='000593':
                #    embed()
                frame_labels = gt_label_query[vid][fid]
                label_ = '"' +  ','.join(frame_labels) + '"'
        else:
            label_ = '""'
        
        labels.append(label_)
df = pd.DataFrame({
        'original_vido_id':original_vido_id,
        'video_id':video_id,
        'frame_id':frame_id,
        'path':path,
        'labels':labels
    })
df.to_csv(mode + "_aimos.csv", sep='\t', index = False)
