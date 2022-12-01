import json
import copy
import argparse
import os
print("support rootpath ~/", os.path.expanduser("~"))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="split detected graph into Qtype")
	parser.add_argument("--gt_path",default='./GT/')
	parser.add_argument("--det_path", default='./Swin_STTrans_18Epoch/')
	parser.add_argument("--det_type", default='SGDet')
	parser.add_argument("--det_model", default='swin')
	args = parser.parse_args()
	qtype = ['Interaction','Sequence','Prediction','Feasibility']
	dtype = ['train','val','test']

        # get abs path
	args.gt_path = os.path.expanduser(args.gt_path)
	args.det_path = os.path.expanduser(args.det_path)

	save_path = args.det_path + '/' +args.det_type + '/'

	print('gt_path:', args.gt_path)
	print('det_path:', args.det_path)
	print('save_path:', save_path)

	empty_count = 0

	missed_frame = {'train':{},'val':{},'test':{}}

	for d in dtype:
		pre_file = 'STAR_'+d+'_'+ args.det_type.lower() +'_sg_' + args.det_model + '.json'
		pred_sg = json.load(open(args.det_path+'/'+args.det_type+'/'+pre_file))
		for q in qtype:
			empty_count = 0
			QA_New = []
			file = q + '_' + d + '.json'
			print(file, 'Start')
			QA = json.load(open(args.gt_path+file))
			for qa in QA:
				new_qa = copy.deepcopy(qa)
				video_id = qa['video_id']
				situation = qa['situations']
				new_situation = {}
				for f in situation:
					meta = video_id + '.mp4/' + f + '.png'
					if meta not in pred_sg:
						if video_id in missed_frame[d]:
							missed_frame[d][video_id].append(f)
						else:
							missed_frame[d][video_id]=[f]
						#print(meta)
						continue
					new_situation[f] = copy.deepcopy(pred_sg[meta])
					new_situation[f]['actions'] = situation[f]['actions']
					
				if len(new_situation.keys())==0:
					empty_count += 1
					#print(qa['question_id'], 'is empty, use gt')
					new_qa['situations'] = situation
				else:
					new_qa['situations'] = new_situation
				QA_New.append(new_qa)

			save_file = save_path + file
			with open(save_file, 'w') as f:
				f.write(json.dumps(QA_New))
			#print(file, 'Done')
			print(file,'empty graph QA num' ,empty_count)

	missed_frame_ = {'train':{},'val':{},'test':{}}
	miss_count = {'train':0,'val':0,'test':0}
	for d in dtype:
		for v in missed_frame[d]:
			missed_frame_[d][v] = list(set(missed_frame[d][v]))
			miss_count[d]+=len(missed_frame_[d][v])
	print(miss_count)

	with open(args.det_path + '/missed_frame_' + args.det_type.lower() + '.json', 'w') as f:
		f.write(json.dumps(missed_frame_))
