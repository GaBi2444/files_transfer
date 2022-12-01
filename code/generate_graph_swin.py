import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import numpy as np
np.set_printoptions(precision=4)
import copy
import torch

from dataloader.star import STAR, cuda_collate_fn
from utils_sgdet import generate_scene_graph
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.object_detector_swin import detector
from lib.sttran_swin import STTran
from tqdm import tqdm
import json
from mmdetection.tools.STT_test import swin_init

conf = Config()
print('The CKPT saved here:', conf.save_path)
if not os.path.exists(conf.save_path):
    os.mkdir(conf.save_path)
for i in conf.args:
    print(i,':', conf.args[i])

STAR_dataset = STAR(qa_path=conf.data_path + '/Question_Answer_SituationGraph/', split=conf.split, mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                filter_small_box=False if conf.mode == 'predcls' else True, generate_mode=True)
dataloader = torch.utils.data.DataLoader(STAR_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)
swin = swin_init()
gpu_device = torch.device('cuda:0')
object_detector = detector(model = swin, train=False, object_classes=STAR_dataset.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
object_detector.eval()


model = STTran(mode=conf.mode,
               attention_class_num=len(STAR_dataset.attention_relationships),
               spatial_class_num=len(STAR_dataset.spatial_relationships),
               contact_class_num=len(STAR_dataset.contacting_relationships),
               obj_classes=STAR_dataset.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer).to(device=gpu_device)

model.eval()

ckpt = torch.load(conf.model_path, map_location=gpu_device)
model.load_state_dict(ckpt['state_dict'], strict=False)
print('*'*50)
print('CKPT {} is loaded'.format(conf.model_path))
#
evaluator1 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=STAR_dataset.object_classes,
    AG_all_predicates=STAR_dataset.relationship_classes,
    AG_attention_predicates=STAR_dataset.attention_relationships,
    AG_spatial_predicates=STAR_dataset.spatial_relationships,
    AG_contacting_predicates=STAR_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='with')

evaluator2 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=STAR_dataset.object_classes,
    AG_all_predicates=STAR_dataset.relationship_classes,
    AG_attention_predicates=STAR_dataset.attention_relationships,
    AG_spatial_predicates=STAR_dataset.spatial_relationships,
    AG_contacting_predicates=STAR_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='semi', semithreshold=0.9)

evaluator3 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=STAR_dataset.object_classes,
    AG_all_predicates=STAR_dataset.relationship_classes,
    AG_attention_predicates=STAR_dataset.attention_relationships,
    AG_spatial_predicates=STAR_dataset.spatial_relationships,
    AG_contacting_predicates=STAR_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='no')

result = {}

with torch.no_grad():
    for b, data in enumerate(tqdm(dataloader)):

        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        frame_names = data[5]
        gt_annotation = STAR_dataset.gt_annotations[data[4]]

        try:
            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            pred = model(entry)
        except:
            print('No Element in AG SG, Skip')
            continue

        pred_entrys = evaluator1.detect_scene_graph(gt_annotation, dict(pred))
        
        for fid, pre_entry in zip(frame_names, pred_entrys):
            try:
                sg = generate_scene_graph(pre_entry)
                result[fid] = sg
            except:
                print(fid ,'No Element in STAR SG, Skip')
                continue

        # if b>20:
        #     break
#         evaluator2.evaluate_scene_graph(gt_annotation, dict(pred))
#         evaluator3.evaluate_scene_graph(gt_annotation, dict(pred))

save_path = os.path.join(conf.save_path, 'STAR_' + conf.split + '_'+ conf.mode + '_sg_swin.json')
with open(save_path,'w') as f:
    f.write(json.dumps(result))

print('Detected SG Num:', len(result.keys()))

# print('-------------------------with constraint-------------------------------')
# evaluator1.print_stats()
# print('-------------------------semi constraint-------------------------------')
# evaluator2.print_stats()
# print('-------------------------no constraint-------------------------------')
# evaluator3.print_stats()
