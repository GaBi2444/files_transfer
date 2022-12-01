import json
import argparse

def split_by_qtype(QA):
    qa_dict = {'Interaction':[],'Sequence':[],'Prediction':[],'Feasibility':[]}
    for qa in QA:
        qa_dict[qa['question_id'].split('_')[0]].append(qa)
    return qa_dict

def save_json(qa,save_path,qtype,dtype):
    save_file = save_path + qtype + '_' + dtype + '.json'
    with open(save_file,'w') as f:
        f.write(json.dumps(qa))
        
def split(args):
    qa_train = json.load(open(args.qa_path+'STAR_train.json'))
    qa_val = json.load(open(args.qa_path+'STAR_val.json'))
    qa_test = json.load(open(args.qa_path+'STAR_test.json'))
    dataset_dict = {'train': split_by_qtype(qa_train),'val': split_by_qtype(qa_val),'test': split_by_qtype(qa_test)}
    for dtype in dataset_dict:
        for qtype in dataset_dict[dtype]:
            save_json(dataset_dict[dtype][qtype],args.save_path,qtype,dtype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Easy to split QA and save to jsonl')
    parser.add_argument('--qa_path',type=str, default='./')
    parser.add_argument('--save_path',type=str, default= './')
    args = parser.parse_args()
    split(args)