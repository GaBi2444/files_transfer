import json
from IPython import embed
mode = ['Feasibility', 'Interaction', 'Sequence', 'Prediction']
#atm_output = []
gt = []
det_act_path = "/home/bowu/data/STAR_feature/ActRecog/MViTv2/STAR_test_temporal_act.json"
det_act = json.load(open(det_act_path))
#embed() 
for j in range(len(mode)):
    atm_output = json.load(open("../exp/checkpoints_10Epoch_temporal_act/" + mode[j]+ "_GT_Sem/star_" + mode[j] + "_action_transition_model.json",'rb'))
    atm_backup = atm_output
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
    for i,q in enumerate(atm_output):
        qid = q[0]
        if qid not in det_act: 
            continue
            not_match += 1
        for fid in q[1]:
            correct_num = 0
            #f_actions = q[1][fid]['actions']
            #if fid not in gt_query[qid]: continue
            #gt_acts = gt_query[qid][fid]
            #gt_num = len(gt_acts)
            #topk = gt_num
            #f_actions = q[1][fid]['actions'][:topk]
            #if gt_num ==0: continue
            #atm_output[j][i][1][fid]['actions'] = gt_acts
            
            if fid in det_act[qid]: 
                det_acts = []
                acts_label = det_act[qid][fid]['pred_actions'][:2]
                for label in acts_label:
                    det_acts.append('a' + str(label).zfill(3))
                atm_output[i][1][fid]['actions'] = det_acts
            #for act in f_actions:
                #if act in gt_acts:
                
                   # correct_num += 1
            #acc = correct_num /gt_num
            #if acc > 1:
            #    print("gt: " + str(gt_acts) + "pred: " + str(f_actions))
            #    print("correct_num: " + str(correct_num) + "gt_num: " + str(gt_num))
            #qid_acc.append(min(1,acc))
    #print(qid_acc)
    #final_acc = sum(qid_acc) / len(qid_acc)
    #print("AMT mode " + mode[j] + " output acc: " + str(final_acc))
    save_gt = True
    print(not_match)
    #embed()
    if save_gt:
        with open("../exp/InitGraph_w_act/" + mode[j] + "_GT_Sem/star_" + mode[j] + "_action_transition_model.json",'w') as f:
            json.dump(atm_output, f)
