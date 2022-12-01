import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import cv2
import os

from lib.funcs import assign_relations
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from fasterRCNN.lib.model.faster_rcnn.resnet import resnet
from fasterRCNN.lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from fasterRCNN.lib.model.roi_layers import nms
from IPython import embed
from torch.nn import functional as F
from mmdetection.tools.STT_test import swin_init
import torchvision
class detector(nn.Module):

    '''first part: object detection (image/video)'''

    def __init__(self, model, train, object_classes, use_SUPPLY, mode='predcls'):
        super(detector, self).__init__()

        self.is_train = train
        self.use_SUPPLY = use_SUPPLY
        self.object_classes = object_classes
        self.mode = mode
        self.model = model
        self.fasterRCNN = resnet(classes=self.object_classes, num_layers=101, pretrained=False, class_agnostic=False)
        self.fasterRCNN.create_architecture()
        checkpoint = torch.load('fasterRCNN/models/faster_rcnn_ag.pth')
        self.fasterRCNN.load_state_dict(checkpoint['model'])

        self.ROI_Align = copy.deepcopy(self.fasterRCNN.RCNN_roi_align)
        self.RCNN_Head = copy.deepcopy(self.fasterRCNN._head_to_tail)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all):

        
        ##todo
        
        import numpy as np
        datapath = '/gpfs/u/home/NSVR/NSVRbowu/scratch/data/STAR_feature/object_detection_swin_features'
        frame_names = []
        for sg_frame in gt_annotation:
            frame_name = sg_frame[1]['metadata']['tag']
            vid, obj_cat, fid = frame_name.split('/')
            vid, _ = vid.split('.')
            frame_names.append(fid)
        frame_info = []
        bboxes = []
        frames_feature = []
        bboxes_feat = []
        SWIN_FEATURES = torch.tensor([]).cuda(0)
        SWIN_BASE_FEATURES = torch.tensor([]).cuda(0)
        frame_index = 0 
        for fid in frame_names:
            
            frame_path = datapath + '/' + vid + '/' + fid + '/'
            #if not os.path.exists(frame_path):
            #   continue
            datanames = os.listdir(frame_path)
            list = []
            objs_info_sg_frame =  []
            #print(datanames)
            for i in range(len(datanames)):
                if datanames[i] == fid +'.npz':
                #print(frame_path + datanames[i])
                    sg_frame_feat = np.load(frame_path + datanames[i] , allow_pickle=True)
                    sg_frame_feat_tensor = torch.tensor(sg_frame_feat['arr_0'])
                    #todo
        
                    #print(sg_frame_feat_tensor.shape)
                    
                    sg_frame_feat_tensor = sg_frame_feat_tensor.float()
                    query_size_h=sg_frame_feat_tensor.shape[2]
                    query_size_w=sg_frame_feat_tensor.shape[3]
                    #print(query_size)
                    if query_size_h == 336:
                        sg_frame_feat_tensor = F.interpolate(sg_frame_feat_tensor, size = [ 67, 38], mode="bilinear")
                    if query_size_h == 304:
                        sg_frame_feat_tensor = F.interpolate(sg_frame_feat_tensor, size = [ 57, 38], mode="bilinear")
                    if query_size_h == 272:
                        sg_frame_feat_tensor = F.interpolate(sg_frame_feat_tensor, size = [ 50, 38], mode="bilinear")
                    if query_size_w == 336:
                        sg_frame_feat_tensor = F.interpolate(sg_frame_feat_tensor, size = [ 38, 67], mode="bilinear")
                    if query_size_w == 304:
                        sg_frame_feat_tensor = F.interpolate(sg_frame_feat_tensor, size = [ 38, 57], mode="bilinear")
                    if query_size_w == 272:
                        sg_frame_feat_tensor = F.interpolate(sg_frame_feat_tensor, size = [ 38, 50], mode="bilinear")
                    sg_frame_feat_tensor = sg_frame_feat_tensor.cuda(0)
                    #print(sg_frame_feat_tensor.shape)
                    frames_feature.append(sg_frame_feat_tensor)
                    SWIN_BASE_FEATURES = torch.cat((SWIN_BASE_FEATURES, sg_frame_feat_tensor), 0)
                else:
                    obj_info_all = np.load(frame_path + datanames[i], allow_pickle=True)
                    obj_info = obj_info_all['obj_info'][()]
                    #print(obj_info['bbox'].shape)
                    obj_info['bbox'][2] = obj_info['bbox'][0]+obj_info['bbox'][2]
                    obj_info['bbox'][3] = obj_info['bbox'][1]+obj_info['bbox'][3]
                    obj_info['bbox'].insert(0,frame_index)
                    #print(obj_info['bbox'].shape)
                    obj_feat = obj_info_all['obj_feat']
                    obj_feat = torch.tensor(obj_feat)
                    obj_feat = obj_feat.cuda(0)
                    SWIN_FEATURES = torch.cat((SWIN_FEATURES, obj_feat.unsqueeze(0)), 0)
                    bboxes_feat.append(torch.tensor(obj_feat))
                    objs_info_sg_frame.append(obj_info)
                    bboxes.append(obj_info)
            frame_index += 1
        SWIN_BBOXES = torch.tensor([]).cuda(0)
        SWIN_LABELS = torch.tensor([], dtype=torch.int64).cuda(0)
        SWIN_SCORES = torch.tensor([]).cuda(0)
        for sg_bbox in bboxes:
            swin_bbox = torch.tensor(sg_bbox['bbox'])
            swin_label = torch.tensor(sg_bbox['label'])
            swin_score = torch.tensor(sg_bbox['score'])
            swin_bbox=swin_bbox.cuda(0)
            swin_label=swin_label.cuda(0)
            swin_score=swin_score.cuda(0)
            SWIN_BBOXES = torch.cat((SWIN_BBOXES, swin_bbox.unsqueeze(0)), 0)
            SWIN_LABELS = torch.cat((SWIN_LABELS, swin_label.unsqueeze(0)), 0)
            SWIN_SCORES = torch.cat((SWIN_SCORES, swin_score.unsqueeze(0)), 0)
        #embed()
        #frame_feature
        #print(FINAL_BASE_FEATURES.shape)

        if self.mode == 'sgdet':
            counter = 0
            counter_image = 0

            
            # prediction = {'FINAL_BBOXES': FINAL_BBOXES, 'FINAL_LABELS': FINAL_LABELS, 'FINAL_SCORES': FINAL_SCORES,
            #               'FINAL_FEATURES': FINAL_FEATURES, 'FINAL_BASE_FEATURES': FINAL_BASE_FEATURES}
            prediction = {'FINAL_BBOXES': SWIN_BBOXES, 'FINAL_LABELS': SWIN_LABELS, 'FINAL_SCORES': SWIN_SCORES,
                          'FINAL_FEATURES': SWIN_FEATURES, 'FINAL_BASE_FEATURES': SWIN_BASE_FEATURES}
            FINAL_BBOXES = SWIN_BBOXES
            FINAL_LABELS = SWIN_LABELS
            FINAL_SCORES = SWIN_SCORES
            FINAL_FEATURES = SWIN_FEATURES
            FINAL_BASE_FEATURES = SWIN_BASE_FEATURES
            
        
            if self.is_train:

                DETECTOR_FOUND_IDX, GT_RELATIONS, SUPPLY_RELATIONS, assigned_labels = assign_relations(prediction, gt_annotation, assign_IOU_threshold=0.5)

                if self.use_SUPPLY:
                    # supply the unfounded gt boxes by detector into the scene graph generation training
                    FINAL_BBOXES_X = torch.tensor([]).cuda(0)
                    FINAL_LABELS_X = torch.tensor([], dtype=torch.int64).cuda(0)
                    FINAL_SCORES_X = torch.tensor([]).cuda(0)
                    FINAL_FEATURES_X = torch.tensor([]).cuda(0)
                    assigned_labels = torch.tensor(assigned_labels, dtype=torch.long).to(FINAL_BBOXES_X.device)

                    for i, j in enumerate(SUPPLY_RELATIONS):
                        if len(j) > 0:
                            unfound_gt_bboxes = torch.zeros([len(j), 5]).cuda(0)
                            unfound_gt_classes = torch.zeros([len(j)], dtype=torch.int64).cuda(0)
                            one_scores = torch.ones([len(j)], dtype=torch.float32).cuda(0)  # probability
                            for m, n in enumerate(j):
                                # if person box is missing or objects
                                im_info.cuda(0)
                                if 'bbox' in n.keys():
                                    unfound_gt_bboxes[m, 1:] = torch.tensor(n['bbox']).cuda(0) * im_info[
                                        i, 2]  # don't forget scaling!
                                    unfound_gt_classes[m] = n['class']
                                else:
                                    # here happens always that IOU <0.5 but not unfounded
                                    unfound_gt_bboxes[m, 1:] = torch.tensor(n['person_bbox']).cuda(0) * im_info[
                                        i, 2]  # don't forget scaling!
                                    unfound_gt_classes[m] = 1  # person class index

                            DETECTOR_FOUND_IDX[i] = np.concatenate((DETECTOR_FOUND_IDX[i],
                                                                         np.arange(
                                                                             start=int(sum(FINAL_BBOXES[:, 0] == i)),
                                                                             stop=int(
                                                                                 sum(FINAL_BBOXES[:, 0] == i)) + len(
                                                                                 SUPPLY_RELATIONS[i]))), axis=0).astype(
                                'int64').tolist()

                            GT_RELATIONS[i].extend(SUPPLY_RELATIONS[i])

                            # compute the features of unfound gt_boxes
                            #todo
                            
                            pooled_feat_align = torchvision.ops.roi_align(FINAL_BASE_FEATURES[i].unsqueeze(0),
                                                                         unfound_gt_bboxes.cuda(0), output_size=(7, 7), spatial_scale = 1.0/16.0)
                            _cls_score, _pred_bbox, pooled_feat = self.model.module.roi_head.bbox_head(pooled_feat_align)


                            unfound_gt_bboxes[:, 0] = i
                            unfound_gt_bboxes[:, 1:] = unfound_gt_bboxes[:, 1:] / im_info[i, 2]
                            FINAL_BBOXES_X = torch.cat(
                                (FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i], unfound_gt_bboxes))
                            FINAL_LABELS_X = torch.cat((FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i],
                                                        unfound_gt_classes))  # final label is not gt!
                            FINAL_SCORES_X = torch.cat(
                                (FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i], one_scores))
                            FINAL_FEATURES_X = torch.cat(
                                (FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i], pooled_feat))
                        else:
                            FINAL_BBOXES_X = torch.cat((FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i]))
                            FINAL_LABELS_X = torch.cat((FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i]))
                            FINAL_SCORES_X = torch.cat((FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i]))
                            FINAL_FEATURES_X = torch.cat((FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i]))

                FINAL_DISTRIBUTIONS = self.model.module.roi_head.bbox_head.fc_cls(FINAL_FEATURES_X)[:, :-1]
            
                global_idx = torch.arange(start=0, end=FINAL_BBOXES_X.shape[0])  # all bbox indices

                im_idx = []  # which frame are the relations belong to
                pair = []
                a_rel = []
                s_rel = []
                c_rel = []
                for i, j in enumerate(DETECTOR_FOUND_IDX):

                    for k, kk in enumerate(GT_RELATIONS[i]):
                        if 'person_bbox' in kk.keys():
                            kkk = k
                            break
                    localhuman = int(global_idx[FINAL_BBOXES_X[:, 0] == i][kkk])

                    for m, n in enumerate(j):
                        if 'class' in GT_RELATIONS[i][m].keys():
                            im_idx.append(i)

                            pair.append([localhuman, int(global_idx[FINAL_BBOXES_X[:, 0] == i][int(n)])])

                            a_rel.append(GT_RELATIONS[i][m]['attention_relationship'].tolist())
                            s_rel.append(GT_RELATIONS[i][m]['spatial_relationship'].tolist())
                            c_rel.append(GT_RELATIONS[i][m]['contacting_relationship'].tolist())

                pair = torch.tensor(pair).cuda(0)
                im_idx = torch.tensor(im_idx, dtype=torch.float).cuda(0)
                union_boxes = torch.cat((im_idx[:, None],
                                         torch.min(FINAL_BBOXES_X[:, 1:3][pair[:, 0]],
                                                   FINAL_BBOXES_X[:, 1:3][pair[:, 1]]),
                                         torch.max(FINAL_BBOXES_X[:, 3:5][pair[:, 0]],
                                                   FINAL_BBOXES_X[:, 3:5][pair[:, 1]])), 1)

                union_boxes[:, 1:] = union_boxes[:, 1:] * im_info[0, 2]
                union_feat = torchvision.ops.roi_align(FINAL_BASE_FEATURES,
                                    union_boxes, output_size=(7, 7), spatial_scale = 1.0/16.0)

                pair_rois = torch.cat((FINAL_BBOXES_X[pair[:,0],1:],FINAL_BBOXES_X[pair[:,1],1:]), 1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                entry = {'boxes': FINAL_BBOXES_X,
                         'labels': FINAL_LABELS_X,
                         'scores': FINAL_SCORES_X,
                         'distribution': FINAL_DISTRIBUTIONS,
                         'im_idx': im_idx,
                         'pair_idx': pair,
                         'features': FINAL_FEATURES_X,
                         'union_feat': union_feat,
                         'spatial_masks': spatial_masks,
                         'attention_gt': a_rel,
                         'spatial_gt': s_rel,
                         'contacting_gt': c_rel}

                return entry

            else:
                FINAL_DISTRIBUTIONS = self.model.module.roi_head.bbox_head.fc_cls(FINAL_FEATURES)[:, :-1]
            
                FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                PRED_LABELS = PRED_LABELS + 1

                entry = {'boxes': FINAL_BBOXES,
                         'scores': FINAL_SCORES,
                         'distribution': FINAL_DISTRIBUTIONS,
                         'pred_labels': PRED_LABELS,
                         'features': FINAL_FEATURES,
                         'fmaps': FINAL_BASE_FEATURES,
                         'im_info': im_info[0, 2]}

                return entry
        else:
            # how many bboxes we have
            bbox_num = 0

            im_idx = []  # which frame are the relations belong to
            pair = []
            a_rel = []
            s_rel = []
            c_rel = []

            for i in gt_annotation:
                bbox_num += len(i)
            FINAL_BBOXES = torch.zeros([bbox_num,5], dtype=torch.float32).cuda(0)
            FINAL_LABELS = torch.zeros([bbox_num], dtype=torch.int64).cuda(0)
            FINAL_SCORES = torch.ones([bbox_num], dtype=torch.float32).cuda(0)
            HUMAN_IDX = torch.zeros([len(gt_annotation),1], dtype=torch.int64).cuda(0)

            bbox_idx = 0
            for i, j in enumerate(gt_annotation):
                for m in j:
                    if 'person_bbox' in m.keys():
                        FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['person_bbox'][0])
                        FINAL_BBOXES[bbox_idx, 0] = i
                        FINAL_LABELS[bbox_idx] = 1
                        HUMAN_IDX[i] = bbox_idx
                        bbox_idx += 1
                    else:
                        FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['bbox'])
                        FINAL_BBOXES[bbox_idx, 0] = i
                        FINAL_LABELS[bbox_idx] = m['class']
                        im_idx.append(i)
                        pair.append([int(HUMAN_IDX[i]), bbox_idx])
                        a_rel.append(m['attention_relationship'].tolist())
                        s_rel.append(m['spatial_relationship'].tolist())
                        c_rel.append(m['contacting_relationship'].tolist())
                        bbox_idx += 1
            pair = torch.tensor(pair).cuda(0)
            im_idx = torch.tensor(im_idx, dtype=torch.float).cuda(0)

            counter = 0
            FINAL_BASE_FEATURES = torch.tensor([]).cuda(0)


            FINAL_BASE_FEATURES = SWIN_BASE_FEATURES
            FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] * im_info[0, 2]

            FINAL_FEATURES = torchvision.ops.roi_align(FINAL_BASE_FEATURES, FINAL_BBOXES, output_size=(7, 7), spatial_scale=1.0 / 16.0)
            cls_score, _, FINAL_FEATURES = self.model.module.roi_head.bbox_head(FINAL_FEATURES)


            if self.mode == 'predcls':

                union_boxes = torch.cat((im_idx[:, None], torch.min(FINAL_BBOXES[:, 1:3][pair[:, 0]], FINAL_BBOXES[:, 1:3][pair[:, 1]]),
                                         torch.max(FINAL_BBOXES[:, 3:5][pair[:, 0]], FINAL_BBOXES[:, 3:5][pair[:, 1]])), 1)
                union_feat = torchvision.ops.roi_align(FINAL_BASE_FEATURES, union_boxes, output_size=(7, 7), spatial_scale=1.0 / 16.0)

                FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
                pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                entry = {'boxes': FINAL_BBOXES,
                         'labels': FINAL_LABELS, # here is the groundtruth
                         'scores': FINAL_SCORES,
                         'im_idx': im_idx,
                         'pair_idx': pair,
                         'human_idx': HUMAN_IDX,
                         'features': FINAL_FEATURES,
                         'union_feat': union_feat,
                         'union_box': union_boxes,
                         'spatial_masks': spatial_masks,
                         'attention_gt': a_rel,
                         'spatial_gt': s_rel,
                         'contacting_gt': c_rel
                        }

                return entry
            elif self.mode == 'sgcls':
                if self.is_train:

                    FINAL_DISTRIBUTIONS = cls_score[:,:-1]
                    FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                    PRED_LABELS = PRED_LABELS + 1

                    union_boxes = torch.cat(
                        (im_idx[:, None], torch.min(FINAL_BBOXES[:, 1:3][pair[:, 0]], FINAL_BBOXES[:, 1:3][pair[:, 1]]),
                         torch.max(FINAL_BBOXES[:, 3:5][pair[:, 0]], FINAL_BBOXES[:, 3:5][pair[:, 1]])), 1)
                    union_feat = torchvision.ops.roi_align(FINAL_BASE_FEATURES, union_boxes, output_size=(7, 7),
                                                           spatial_scale=1.0 / 16.0)
                    FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
                    pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]),
                                          1).data.cpu().numpy()
                    spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                    entry = {'boxes': FINAL_BBOXES,
                             'labels': FINAL_LABELS,  # here is the groundtruth
                             'scores': FINAL_SCORES,
                             'distribution': FINAL_DISTRIBUTIONS,
                             'pred_labels': PRED_LABELS,
                             'im_idx': im_idx,
                             'pair_idx': pair,
                             'human_idx': HUMAN_IDX,
                             'features': FINAL_FEATURES,
                             'union_feat': union_feat,
                             'union_box': union_boxes,
                             'spatial_masks': spatial_masks,
                             'attention_gt': a_rel,
                             'spatial_gt': s_rel,
                             'contacting_gt': c_rel}

                    return entry
                else:
                    FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]

                    FINAL_DISTRIBUTIONS = cls_score[:,:-1]
                    FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                    PRED_LABELS = PRED_LABELS + 1

                    entry = {'boxes': FINAL_BBOXES,
                             'labels': FINAL_LABELS,  # here is the groundtruth
                             'scores': FINAL_SCORES,
                             'distribution': FINAL_DISTRIBUTIONS,
                             'pred_labels': PRED_LABELS,
                             'im_idx': im_idx,
                             'pair_idx': pair,
                             'human_idx': HUMAN_IDX,
                             'features': FINAL_FEATURES,
                             'attention_gt': a_rel,
                             'spatial_gt': s_rel,
                             'contacting_gt': c_rel,
                             'fmaps': FINAL_BASE_FEATURES,
                             'im_info': im_info[0, 2]}

                    return entry

