import argparse
import tensorflow as tf
import json
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import sys
from scipy import stats
import time

import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../../'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../models'))
import provider
from utils.test_utils import *
from models import model

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default="1", help='GPU to use [default: GPU 1]')
parser.add_argument('--verbose', action='store_true', help='if specified, use depthconv')
parser.add_argument('--input_list', type=str, default='data/test_hdf5_file_list.txt', help='Validation data list')
parser.add_argument('--restore_dir', type=str, default='checkpoint/stanford_ins_seg_groupmask11_fromgroup_recipweight_nopow2_lr4', help='Directory that stores all training logs and trained models')

FLAGS = parser.parse_args()

# DEFAULT SETTINGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
PRETRAINED_MODEL_PATH = os.path.join(FLAGS.restore_dir,'trained_models/')

RESTORE_DIR = FLAGS.restore_dir
gpu_to_use = 0
OUTPUT_DIR = os.path.join(FLAGS.restore_dir, 'test_results')
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

output_verbose = FLAGS.verbose  # If true, output all color-coded segmentation obj files

label_bin = np.loadtxt(os.path.join(RESTORE_DIR, 'pergroup_thres.txt'))
min_num_pts_in_group = np.loadtxt(os.path.join(RESTORE_DIR, 'mingroupsize.txt'))

# MAIN SCRIPT
POINT_NUM = 4096  # the max number of points in the all testing data shapes
BATCH_SIZE  = 1
NUM_GROUPS = 50
NUM_CATEGORY = 13

TESTING_FILE_LISTFILE = FLAGS.input_list
test_file_list = provider.getDataFiles(TESTING_FILE_LISTFILE)
len_pts_files = len(test_file_list)

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def predict():
    is_training = False

    with tf.device('/gpu:' + str(gpu_to_use)):
        is_training_ph = tf.placeholder(tf.bool, shape=())

        pointclouds_ph, ptsseglabel_ph, ptsgroup_label_ph, _, _, _ = \
            model.placeholder_inputs(BATCH_SIZE, POINT_NUM, NUM_GROUPS, NUM_CATEGORY)

        net_output = model.get_model(pointclouds_ph, is_training_ph, group_cate_num=NUM_CATEGORY)

        group_mat_label = tf.matmul(ptsgroup_label_ph, tf.transpose(ptsgroup_label_ph, perm=[0, 2, 1])) #BxNxN: (i,j) if i and j in the same group

    # Add ops to save and restore all the variables.

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:

        flog = open(os.path.join(OUTPUT_DIR, 'log.txt'), 'w')

        # Restore variables from disk.

        ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH,os.path.basename(ckptstate.model_checkpoint_path))
            saver.restore(sess, LOAD_MODEL_FILE)
            printout(flog, "Model loaded in file: %s" % LOAD_MODEL_FILE)
        else:
            printout(flog, "Fail to load modelfile: %s" % PRETRAINED_MODEL_PATH)

        total_acc = 0.0
        total_seen = 0

        ious = np.zeros(NUM_CATEGORY)
        totalnums = np.zeros(NUM_CATEGORY)

        tpsins = [[] for itmp in range(NUM_CATEGORY)]#= np.array([]).reshape(0, NUM_CATEGORY)#np.zeros(NUM_CATEGORY)#
        fpsins = [[] for itmp in range(NUM_CATEGORY)]#= np.array([]).reshape(0, NUM_CATEGORY)#np.zeros(NUM_CATEGORY)#

        positive_ins_sgpn = np.zeros(NUM_CATEGORY)
        total_sgpn = np.zeros(NUM_CATEGORY)
        at = 0.5


        for shape_idx in range(len_pts_files):

            cur_train_filename = test_file_list[shape_idx]
            printout(flog, '%d / %d ...' % (shape_idx, len_pts_files))
            printout(flog, 'Loading train file ' + cur_train_filename)

            if not os.path.exists(cur_train_filename):
                continue
            cur_data, cur_group, _, cur_seg = provider.loadDataFile_with_groupseglabel_stanfordindoor(cur_train_filename)

            seg_output = np.zeros_like(cur_seg)
            segrefine_output = np.zeros_like(cur_seg)
            group_output = np.zeros_like(cur_group)

            pts_label_one_hot, pts_label_mask = model.convert_seg_to_one_hot(cur_seg)
            pts_group_label, _ = model.convert_groupandcate_to_one_hot(cur_group)
            num_data = cur_data.shape[0]

            gap = 5e-3
            volume_num = int(1. / gap)+1
            volume = -1* np.ones([volume_num,volume_num,volume_num]).astype(np.int32)
            volume_seg = -1* np.ones([volume_num,volume_num,volume_num]).astype(np.int32)

            intersections = np.zeros(NUM_CATEGORY)
            unions = np.zeros(NUM_CATEGORY)

            for j in range(num_data):
                print ("Processsing: Shape [%d] Block[%d]"%(shape_idx, j))

                pts = cur_data[j,...]

                feed_dict = {
                    pointclouds_ph: np.expand_dims(pts,0),
                    ptsseglabel_ph: np.expand_dims(pts_label_one_hot[j,...],0),
                    ptsgroup_label_ph: np.expand_dims(pts_group_label[j,...],0),
                    is_training_ph: is_training,
                }

                pts_corr_val0, pred_confidence_val0, ptsclassification_val0, pts_corr_label_val0 = \
                    sess.run([net_output['simmat'],
                              net_output['conf'],
                              net_output['semseg'],
                              group_mat_label],
                              feed_dict=feed_dict)

                seg = cur_seg[j,...]
                ins = cur_group[j,...]

                pts_corr_val = np.squeeze(pts_corr_val0[0]) #NxG
                pred_confidence_val = np.squeeze(pred_confidence_val0[0])
                ptsclassification_val = np.argmax(np.squeeze(ptsclassification_val0[0]),axis=1)

                seg = np.squeeze(seg)
                # print label_bin

                groupids_block, refineseg, group_seg = GroupMerging(pts_corr_val, pred_confidence_val, ptsclassification_val, label_bin)  # yolo_to_groupt(pts_corr_val, pts_corr_label_val0[0], seg,t=5)

                groupids = BlockMerging(volume, volume_seg, pts[:,6:], groupids_block.astype(np.int32), group_seg, gap)

                seg_output[j,:] = ptsclassification_val
                segrefine_output[j,:] = refineseg
                group_output[j,:] = groupids
                total_acc += float(np.sum(ptsclassification_val==seg))/ptsclassification_val.shape[0]
                total_seen += 1
                # if output_verbose:
                #     output_color_point_cloud(pts[:, 6:], groupids.astype(np.int32),
                #                              os.path.join(OUTPUT_DIR, '%d_%d_grouppred.obj' % (shape_idx,j)))

            ###### Evaluation
            ### Instance Segmentation
            ## Pred
            group_pred = group_output.reshape(-1)
            seg_pred = seg_output.reshape(-1)
            pts = cur_data.reshape([-1, 9])

            # filtering
            x = (pts[:, 6] / gap).astype(np.int32)
            y = (pts[:, 7] / gap).astype(np.int32)
            z = (pts[:, 8] / gap).astype(np.int32)
            for i in range(group_pred.shape[0]):
                if volume[x[i], y[i], z[i]] != -1:
                    group_pred[i] = volume[x[i], y[i], z[i]]

            seg_gt = cur_seg.reshape(-1)
            un = np.unique(group_pred)
            pts_in_pred = [[] for itmp in range(NUM_CATEGORY)]
            group_pred_final = -1 * np.ones_like(group_pred)
            grouppred_cnt = 0

            for ig, g in enumerate(un): #each object in prediction
                if g == -1:
                    continue
                tmp = (group_pred == g)
                sem_seg_g = int(stats.mode(seg_pred[tmp])[0])
                if np.sum(tmp) > 0.25 * min_num_pts_in_group[sem_seg_g]:
                    pts_in_pred[sem_seg_g] += [tmp]
                    group_pred_final[tmp] = grouppred_cnt
                    grouppred_cnt += 1
                    # savepts = pts[tmp, 6:]
                    # savegroup = group_pred[tmp]
                    # if output_verbose:
                    #         output_color_point_cloud(savepts, savegroup.astype(np.int32),
                    #                                  os.path.join(OUTPUT_DIR, '%d_%d_group.obj' % (sem_seg_g, ig)))

            if output_verbose:
                output_color_point_cloud(pts[:, 6:], seg_pred.astype(np.int32),
                                         os.path.join(OUTPUT_DIR, '%d_segpred.obj' % (shape_idx)))
                output_color_point_cloud(pts[:, 6:], group_pred_final.astype(np.int32),
                                         os.path.join(OUTPUT_DIR, '%d_grouppred.obj' % (shape_idx)))


            ## GT
            group_gt = cur_group.reshape(-1)
            un = np.unique(group_gt)
            pts_in_gt = [[] for itmp in range(NUM_CATEGORY)]
            for ig, g in enumerate(un):
                tmp = (group_gt == g)
                sem_seg_g = int(stats.mode(seg_pred[tmp])[0])
                pts_in_gt[sem_seg_g] += [tmp]
                total_sgpn[sem_seg_g] += 1

            for i_sem in range(NUM_CATEGORY):
                tp = [0.] * len(pts_in_pred[i_sem])
                fp = [0.] * len(pts_in_pred[i_sem])
                gtflag = np.zeros(len(pts_in_gt[i_sem]))

                for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                    ovmax = -1.

                    for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                        union = (ins_pred | ins_gt)
                        intersect = (ins_pred & ins_gt)
                        iou = float(np.sum(intersect)) / np.sum(union)

                        if iou > ovmax:
                            ovmax = iou
                            igmax = ig

                    if ovmax >= at:
                        if gtflag[igmax] == 0:
                            tp[ip] = 1 # true
                            gtflag[igmax] = 1
                        else:
                            fp[ip] = 1 # multiple det
                    else:
                        fp[ip] = 1 # false positive

                tpsins[i_sem] += tp
                fpsins[i_sem] += fp


            ### Semantic Segmentation

            un, indices = np.unique(seg_gt, return_index=True)
            for segid in un:
                intersect = np.sum((seg_pred == segid) & (seg_gt == segid))
                union = np.sum((seg_pred == segid) | (seg_gt == segid))
                intersections[segid] += intersect
                unions[segid] += union
            iou = intersections / unions
            for i_iou, iou_ in enumerate(iou):
                if not np.isnan(iou_):
                    ious[i_iou] += iou_
                    totalnums[i_iou] += 1

        ap = np.zeros(NUM_CATEGORY)
        for i_sem in range(NUM_CATEGORY):
            ap[i_sem], _, _ = eval_3d_perclass(tpsins[i_sem], fpsins[i_sem], total_sgpn[i_sem])

        print 'Instance Segmentation AP:', ap
        print 'Instance Segmentation mAP:', np.mean(ap)
        print 'Semantic Segmentation IoU:', ious / totalnums
        print 'Semantic Segmentation Acc: %f' , total_acc/total_seen

with tf.Graph().as_default():
    predict()
