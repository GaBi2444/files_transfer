import json
from IPython import embed
mode = ['Feasibility', 'Interaction', 'Sequence', 'Prediction']
atm_output = []
gt = []
for j in range(len(mode)):
    atm_output.append(json.load(open("../exp/checkpoints_10Epoch_temporal_act/" + mode[j]+ "_GT_Sem/star_" + mode[j] + "_action_transition_model.json",'rb')))
    gt.append(json.load(open("/home/bowu/data/STAR/Question_Answer_SituationGraph/GT/" + mode[j] + "_test.json",'rb')))

    gt_query = {}
    for q_gt in gt:
        qid = q_gt['question_id']
        frames_act = {}
        for fid in q_gt['situations']:
            actions = q_gt['situations'][fid]['actions']
            frames_act[fid] = actions
        gt_query[qid] = frames_act
    qid_acc = []
    for i,q in enumerate(atm_output[j]):
        qid = q[0]
        for fid in q[1]:
            correct_num = 0
            f_actions = q[1][fid]['actions']
            if fid not in gt_query[qid]: continue
            gt_acts = gt_query[qid][fid]
            gt_num = len(gt_acts)
            if gt_num ==0: continue
            atm_output[j][i][1][fid]['actions'] = gt_acts
            for act in f_actions:
                if act in gt_acts:
                
                    correct_num += 1
            acc = correct_num /gt_num
            #if acc > 1:
            #    print("gt: " + str(gt_acts) + "pred: " + str(f_actions))
            #    print("correct_num: " + str(correct_num) + "gt_num: " + str(gt_num))
            qid_acc.append(min(1,acc))
    #print(qid_acc)
    final_acc = sum(qid_acc) / len(qid_acc)
    print("AMT mode " + mode[j] + " output acc: " + str(final_acc))
    save_gt = False
    if save_gt:
        with open('../exp/GT_test/Feasibility_GT_Sem/star_Feasibility_action_transition_model.json','w') as f:
            json.dump(atm_output[j], f)
