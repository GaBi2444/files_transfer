import os
import pickle
import cv2
import csv
import numpy as np
import pandas as pd


def load_annotations(annotation_dir):
    with open(os.path.join(annotation_dir, 'object_bbox_and_relationship.pkl'), 'rb') as f:
        object_anno = pickle.load(f)

    # with open(os.path.join(annotation_dir, 'person_bbox.pkl'), 'rb') as f:
    #     person_anno = pickle.load(f)
    #
    # frame_list = []
    # with open(os.path.join(annotation_dir, 'frame_list.txt'), 'r') as f:
    #     for frame in f:
    #         frame_list.append(frame.rstrip('\n'))

    # return object_anno, person_anno, frame_list
    return object_anno


# split annotation into {Video_No:{Frame_No}} from
def anno_data_split(object_anno):
    anno_video_split = {}
    for k in object_anno.keys():
        s = k.split('/')
        video_id = s[0][:5]
        frame_id = s[1][:6]
        if not anno_video_split.__contains__(video_id):
            anno_video_split[video_id] = {frame_id: object_anno[k]}
        else:
            anno_video_split[video_id][frame_id] = object_anno[k]
    return anno_video_split

def read_video(file_path):
    video_cap = cv2.VideoCapture(file_path)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        video_FPS = video_cap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        video_FPS = video_cap.get(cv2.CAP_PROP_FPS)
    return video_FPS


def load_csv(csv_dir):
    data = None
    for filename in ["/Charades_v1_test.csv", "/Charades_v1_train.csv"]:
        if data is None:
            data = pd.read_csv(csv_dir + filename)
        else:
            tmp_data = pd.read_csv(csv_dir + filename)
            data = pd.concat([data, tmp_data])
            data = data.reset_index(drop=True)
        print('After load_csv(), len:', len(data))
#         with open(csv_dir + filename) as f:
#             f_csv = csv.reader(f)
#             for (i, row) in enumerate(f_csv):
#                 if i == 0:
#                     continue
#                 data.append(row)
    return data


def load_action_mapping(csv_dir):
    # verb_map = {}
    # with open(csv_dir + "/Charades_v1_verbclasses.txt") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         mapping = line.split()
    #         verb_map[mapping[0]] = mapping[1]
    # obj_map = {}
    # with open(csv_dir + "/Charades_v1_objectclasses.txt") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         mapping = line.split()
    #         obj_map[mapping[0]] = mapping[1]
    # dict = {}
    # with open(csv_dir + "/Charades_v1_mapping.txt") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         mapping = line.split()
    #         dict[mapping[0]] = [obj_map[mapping[1]], verb_map[mapping[2]]]
    dict = {}
    with open(csv_dir + "/classes.txt") as f:
        lines = f.readlines()
        for line in lines:
            tag = line[0:4]
            description = line[5:-1]
            dict[tag] = description
    return dict


def load_obj_mapping(csv_dir):
    map = {}
    with open(csv_dir + "/Charades_v1_mapping.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            map[mapping[0]] = mapping[1]
    obj_map = {}
    with open(csv_dir + "/Charades_v1_objectclasses.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            obj_map[mapping[0]] = mapping[1]
    for k in map.keys():
        map[k] = obj_map[map[k]]
    return map

def load_verb_mapping(csv_dir):
    map = {}
    with open(csv_dir + "/Charades_v1_mapping.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            map[mapping[0]] = mapping[2]
    verb_map = {}
    with open(csv_dir + "/Charades_v1_verbclasses.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            verb_map[mapping[0]] = mapping[1]
    for k in map.keys():
        map[k] = verb_map[map[k]]
    return map

def load_verb_expansion_mapping(csv_dir):

    verb_map = {}
    with open(csv_dir + "/verb_expansion.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            verb_map[mapping[0]] = mapping[1] + ' ' + mapping[2]

    return verb_map

def verb_expansion(verb_map,verb_expansion_map):
    for item in verb_map:
        if item in verb_expansion_map.keys():
            verb_map[item] = verb_expansion_map[item]
    return verb_map

# verb_map_ori = load_verb_mapping(CSV_DIR)
# verb_expansion_map = load_verb_expansion_mapping(CSV_DIR)
# VERB_MAP = verb_expansion(verb_map_ori,verb_expansion_map)