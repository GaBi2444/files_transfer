import json
import pickle
import argparse
import pandas as pd
from IPython import embed

filepath = 'cls_star_charades_mapping.csv'
cls_mapping = pd.read_csv(filepath, header = None, encoding='UTF-8')
frame_level_act_graph = json.load((open('frame_level_act_pred.json')))
charade_to_star_dict = {}
for i in range(len(cls_mapping[0])):
    charade_label = int(cls_mapping[0][i][1:])
    star_label = int(cls_mapping[1][i][1:])
    charade_to_star_dict[charade_label] = star_label
for vid in frame_level_act_graph:
    for fid in frame_level_act_graph[vid]:
        charade_label = frame_level_act_graph[vid][fid]['actions_label']
        pop_num = 0
        for i in range(len(charade_label)):
            i = i - pop_num
            if charade_label[i] in charade_to_star_dict:
                star_label = charade_to_star_dict[charade_label[i]]
                frame_level_act_graph[vid][fid]['actions_label'][i] = star_label
            else:
                frame_level_act_graph[vid][fid]['actions_label'].pop(i)
                pop_num += 1
        charade_cand_label = frame_level_act_graph[vid][fid]['actions_candidates']['actions']
        scores = frame_level_act_graph[vid][fid]['actions_candidates']['scores']
        pop_num = 0
        for j in range(len(charade_cand_label)):
            j = j - pop_num
            if charade_cand_label[j] in charade_to_star_dict:
                star_label = charade_to_star_dict[charade_cand_label[j]]
                frame_level_act_graph[vid][fid]['actions_candidates']['actions'][j] = star_label
            else:
                frame_level_act_graph[vid][fid]['actions_candidates']['actions'].pop(j)
                frame_level_act_graph[vid][fid]['actions_candidates']['scores'].pop(j)
                pop_num += 1
        #embed()

print("finish")
filename = "star_frame_level_act_graph.json"
with open(filename, 'w') as file_obj:
    json.dump(frame_level_act_graph, file_obj)
embed()
