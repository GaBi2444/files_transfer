import json
from IPython import embed
mode = ['Feasibility', 'Interaction', 'Sequence', 'Prediction']
#Init_graph = []
gt = []
det_act_path = "/home/bowu/data/STAR_feature/ActRecog/MViTv2/STAR_test_temporal_act.json"
det_act = json.load(open(det_act_path))
#embed()
acc_split = True
if acc_split:
    all_acc = []
    for j in range(len(mode)):
        type_acc = []
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
        not_match = 0
        for i,qid in enumerate(det_act):
            if qid not in gt_query:
                continue
            for fid in det_act[qid]:
                correct_num = 0
                if fid not in gt_query[qid]: continue
                gt_acts = gt_query[qid][fid]
                gt_acts_num = len(gt_acts)
                if gt_acts_num == 0: continue
                det_acts = []
                acts_label = det_act[qid][fid]['pred_actions'][:2]
                for label in acts_label:
                    det_acts.append('a' + str(label).zfill(3))
                for act in det_acts:
                    if act in gt_acts:
                        correct_num += 1
                type_acc.append(correct_num / gt_acts_num)
        qid_acc = sum(type_acc) / len(type_acc)
        print('act acc ' + mode[j] + ' : ' + str(qid_acc))

convert = False
if convert:
    for j in range(len(mode)):
        root_path = "/home/bowu/data/STAR/Question_Answer_SituationGraph/Swin_STTrans_10Epoch/SGDet/"
        Init_graph = json.load(open(root_path + mode[j] + "_test.json",'rb'))
        atm_backup = Init_graph
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
        hypergraphs = []
        for i,video in enumerate(Init_graph):
            hg = []
            qid = video['question_id']
            hg.append(qid)
            situations = video['situations']
            if qid not in det_act: 
                continue
            for fid in situations:
                correct_num = 0
                if fid in det_act[qid]: 
                    det_acts = []
                    acts_label = det_act[qid][fid]['pred_actions'][:2]
                    for label in acts_label:
                        det_acts.append('a' + str(label).zfill(3))
                    situations[fid]['actions'] = det_acts
            hg.append(situations)
        hypergraphs.append(hg)
        save_gt = True
        print(not_match)
        #embed()
        if save_gt:
            with open("../exp/InitGraph_w_act/" + mode[j] + "_GT_Sem/star_" + mode[j] + "_action_transition_model.json",'w') as f:
                json.dump(Init_graph, f)
acc = True
STAR_test_distribution = {}
if acc == True:
    for j in range(len(mode)):
        root_path = "/home/bowu/data/STAR/Question_Answer_SituationGraph/Swin_STTrans_10Epoch/SGDet/"
        Init_graph = json.load(open(root_path + mode[j] + "_test.json",'rb'))
        atm_backup = Init_graph
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
        for i,video in enumerate(Init_graph):
            qid = video['question_id']
            situations = video['situations']
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
                if gt_act_num not in STAR_test_distribution:
                    STAR_test_distribution[gt_act_num] = 1
                else:
                    STAR_test_distribution[gt_act_num] += 1
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
        rel_accuracy = sum(rel_acc) /len(rel_acc)
        obj_accuracy = sum(obj_acc) /len(obj_acc)
        act_accuracy = sum(act_acc) /len(act_acc)
        print(mode[j] + " distribution: " + str(STAR_test_distribution))
        print(mode[j] + " can't find qid num: " + str(not_match))
        print(mode[j] + " Init Graph rel accuracy: " + str(rel_accuracy))
        print(mode[j] + " Init Graph obj accuracy: " + str(obj_accuracy))
        print(mode[j] + " Init Graph action accuracy: " + str(act_accuracy))
                


                
