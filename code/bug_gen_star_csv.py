import json
import pickle
import argparse
import pandas as pd

#original_vido_id / video_id / frame_id / path / labels
def generate_csv(STAR_data, save_path, fps, max_frame_dict, star2charades, star2char=False):
    memory = {}
    original_vido_id = []
    video_id, frame_id = [], []
    path, labels = [], []
    for qa in STAR_data:
        vid = qa['video_id']
        start, end = qa['start'], qa['end']
        start_f, end_f = int(fps[vid+'.mp4']*start) + 1, int(fps[vid+'.mp4']*end) + 1 
        video_meta = vid + '_' + str(int(start_f)) + '_' + str(int(end_f))
        star_kf = qa['situations'].keys()
        star_kf = [int(f) for f in star_kf]

        if video_meta in memory:
            continue
        else:
            memory[video_meta] = 1

        for i in range(start_f,end_f):

            if i > max_frame_dict[vid]:
                continue

            original_vido_id.append(video_meta)
            video_id.append(vid)
            frame_id.append(i)
            path_ = vid + '/' + vid + '-' + str(i).zfill(6) + '.jpg'
            path.append(path_)
            
            if i in star_kf:
                act = qa['situations'][str(i).zfill(6)]['actions']
                if len(act) == 1:
                    if star2char:
                        labels.append(str(int(star2charades[act[0]][1:])))
                    else:
                        labels.append(str(int(act[0][1:])))
                elif len(act) == 0:
                    labels.append("\"")
                else:
                    if star2char:
                        act = [str(int(star2charades[a][1:])) for a in act]
                    else:
                        act = [str(int(a[1:])) for a in act]
                    labels.append(','.join(act))
            else:
                labels.append("\"")

    df = pd.DataFrame({
        'original_vido_id':original_vido_id,
        'video_id':video_id,
        'frame_id':frame_id,
        'path':path,
        'labels':labels,
    })
    df.to_csv(save_path,sep='\t',index=False)

def load_star(star_path, qa_path):

    qa_path_ = star_path + qa_path
    print('='*10, 'Loading STAR Train Dataset', '='*10)
    STAR_train = json.load(open(qa_path_+'STAR_train.json'))
    print('='*10, 'Loading STAR Validation Dataset', '='*10)
    STAR_val = json.load(open(qa_path_+'STAR_val.json'))
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate STAR csv files for SlowFast codebase")
    parser.add_argument("--star_path",default='/nobackup/users/bowu/data/STAR/',help='path to STAR dataset/annotations/frames')
    parser.add_argument("--qa_path", default='Question_Answer_SituationGraph/GT/',help='path to QA dataset')
    parser.add_argument("--anno_path", default='Annotations/',help='path to annotations')
    parser.add_argument("--save_path", default='/nobackup/users/bowu/data/STAR/Situation_Video_Data/',help='path to save csv')
    parser.add_argument("--charades_cls_file", default='/nobackup/users/bowu/data/Charades_v1_480/classes.txt',help='path to charades class file')
    parser.add_argument("--star2char", default=False, type=bool)
    args = parser.parse_args()

    STAR_train, STAR_val, STAR_test = load_star(args.star_path, args.qa_path)
    fps = pickle.load(open(args.star_path + args.anno_path + 'fps', 'rb'))

    charades_cls_map = load_act_cls(args.charades_cls_file)
    star_cls_map = load_act_cls(args.star_path + args.anno_path + 'action_classes.txt')
    max_frame_dict = json.load(open('/nobackup/users/bowu/data/Charades_v1_480/charades_max_frame_ids.json'))

    star2charades = {}
    for cls in star_cls_map:
        star2charades[star_cls_map[cls]] = charades_cls_map[cls]

    print('='*10, 'Generating Test csv', '='*10)
    generate_csv(STAR_test, args.save_path+'test.csv', fps, max_frame_dict, star2charades, args.star2char)
    print('='*10, 'Generating Validation csv', '='*10)
    generate_csv(STAR_val, args.save_path+'val.csv', fps, max_frame_dict, star2charades, args.star2char)
    print('='*10, 'Generating Train csv', '='*10)
    generate_csv(STAR_train, args.save_path+'train.csv', fps, max_frame_dict, star2charades, args.star2char)
    

