import json
from IPython import embed
mode = ['Feasibility', 'Interaction', 'Sequence', 'Prediction']
#Init_graph = []
gt = []
det_act_path = "/home/bowu/data/STAR_feature/ActRecog/MViTv2/STAR_test_temporal_act.json"
det_act = json.load(open(det_act_path))
#embed() 
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
            not_match += 1
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
