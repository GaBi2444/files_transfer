import json
from re import T
from IPython import embed
mode = ['Feasibility', 'Interaction', 'Sequence', 'Prediction']
#atm_output = []
gt = []
det_act_path = "/home/bowu/data/STAR_feature/ActRecog/MViTv2/STAR_test_temporal_act.json"
det_act = json.load(open(det_act_path))
#embed() 
convert = False
if convert:
    for j in range(len(mode)):
        root_path = "../exp/checkpoints_10Epoch_temporal_act/"
        Hyper_graph = json.load(open(root_path + mode[j]+ "_GT_Sem/star_" + mode[j] + "_action_transition_model.json",'rb'))
        gt = json.load(open("/home/bowu/data/STAR/Question_Answer_SituationGraph/GT/" + mode[j] + "_test.json",'rb'))
        gt_query = {}
        for q_gt in gt:
            qid = q_gt['question_id']
            frames_act = {}
            for fid in q_gt['situations']:
                actions = q_gt['situations'][fid]['actions']
                frames_act[fid] = actions
            gt_query[qid] = frames_act
        #embed()
        qid_acc = []
        not_match = 0
        for i,q in enumerate(Hyper_graph):
            qid = q[0]
            if qid not in det_act: 
                continue
                not_match += 1
            for fid in q[1]:
                correct_num = 0

                if fid in det_act[qid]: 
                    det_acts = []
                    acts_label = det_act[qid][fid]['pred_actions'][:2]
                    for label in acts_label:
                        det_acts.append('a' + str(label).zfill(3))
                    Hyper_graph[i][1][fid]['actions'] = det_acts
        save_gt = True
        print(not_match)
        #embed()
        if save_gt:
            with open("../exp/HyperGraph_w_act/" + mode[j] + "_GT_Sem/star_" + mode[j] + "_action_transition_model.json",'w') as f:
                json.dump(Hyper_graph, f)
                
acc = True
if acc == True:
    for j in range(len(mode)):
        root_path = "../exp/checkpoints_10Epoch_temporal_act/"
        Hyper_graph = json.load(open(root_path + mode[j]+ "_GT_Sem/star_" + mode[j] + "_action_transition_model.json",'rb'))
        gt = json.load(open("/home/bowu/data/STAR/Question_Answer_SituationGraph/GT/" + mode[j] + "_test.json",'rb'))

        gt_query = {}
        for q_gt in gt:
            qid = q_gt['question_id']
            frames_info = {}
            for fid in q_gt['situations']:
                frame_ = {}
                rel_labels = q_gt['situations'][fid]['rel_labels']
                bbox_labels = q_gt['situations'][fid]['bbox_labels']
                actions = q_gt['situations'][fid]['actions']
                frame_['rel'] = rel_labels
                frame_['obj'] = bbox_labels
                frame_['act'] = actions
                frames_info[fid] =frame_
            gt_query[qid] = frames_info
        #embed()
        rel_acc = []
        obj_acc = []
        act_acc = []
        not_match = 0
        for i,video in enumerate(Hyper_graph):
            qid = video[0]
            situations = video[1]
            if qid not in gt_query: 
                not_match += 1
                continue
            for fid in situations:
                if fid not in gt_query[qid]: continue
                zero_rel = False
                zero_obj = False
                zero_act = False
                correct_rel_num = 0
                correct_obj_num = 0
                correct_act_num = 0
                gt_rel = gt_query[qid][fid]['rel']
                gt_obj = gt_query[qid][fid]['obj']
                gt_act = gt_query[qid][fid]['act']
                gt_rel_num = len(gt_rel)
                if gt_rel_num == 0: zero_rel = True
                gt_obj_num = len(gt_obj)
                if gt_obj_num == 0: zero_obj = True
                gt_act_num = len(gt_act)
                if gt_act_num == 0: zero_act = True
                init_rel = situations[fid]['rel_labels'][:gt_rel_num]
                init_obj = situations[fid]['bbox_labels'][:gt_obj_num]
                init_act = situations[fid]['actions'][:gt_act_num]
                if not zero_rel:
                    for rel in init_rel:
                        if rel in gt_rel:
                            correct_rel_num += 1
                    rel_acc.append(correct_rel_num/gt_rel_num)
                if not zero_obj:
                    for obj in init_obj:
                        if obj in gt_obj:
                            correct_obj_num += 1
                    obj_acc.append(correct_obj_num/gt_obj_num)
                if not zero_act:
                    for act in init_act:
                        if act in gt_act:
                            correct_act_num += 1
                    act_acc.append(correct_act_num/gt_act_num)
        rel_accuracy = sum(rel_acc) / len(rel_acc)
        obj_accuracy = sum(obj_acc) / len(obj_acc)
        act_accuracy = sum(act_acc) / len(act_acc)
        print(mode[j] + " can't find qid num: " + str(not_match))
        print(mode[j] + " Hyper Graph rel accuracy: " + str(rel_accuracy))
        print(mode[j] + " Hyper Graph obj accuracy: " + str(obj_accuracy))
        print(mode[j] + " Hyper Graph act accuracy: " + str(act_accuracy))