import argparse
import tensorflow as tf
import json
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import sys
from scipy import stats
import time
import h5py

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

parser.add_argument('--gpu', type=str, default="3", help='GPU to use [default: GPU 1]')
parser.add_argument('--verbose', action='store_true', help='if specified, use depthconv')
parser.add_argument('--input_list', type=str, default='/media/hdd2/data/pointnet/stanfordindoor/test_hdf5_file_list5.txt', help='Validation data list')
parser.add_argument('--restore_dir', type=str, default='checkpoint/stanford_ins_seg51', help='Directory that stores all training logs and trained models')

FLAGS = parser.parse_args()

# DEFAULT SETTINGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
PRETRAINED_MODEL_PATH = os.path.join(FLAGS.restore_dir,'trained_models/')

RESTORE_DIR = FLAGS.restore_dir
gpu_to_use = 0
OUTPUT_DIR = os.path.join(FLAGS.restore_dir, 'test_results')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, 'predicted_masks'))

GT_DIR = os.path.join(FLAGS.restore_dir, 'test_gt')
if not os.path.exists(GT_DIR):
    os.makedirs(GT_DIR)
    os.makedirs(os.path.join(GT_DIR, 'predicted_masks'))

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

        ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH,os.path.basename(ckptstate.model_checkpoint_path))
            saver.restore(sess, LOAD_MODEL_FILE)
            print("Model loaded in file: %s" % LOAD_MODEL_FILE)
        else:
            print("Fail to load modelfile: %s" % PRETRAINED_MODEL_PATH)


        for shape_idx in range(len_pts_files):

            cur_train_filename = test_file_list[shape_idx]

            if not os.path.exists(cur_train_filename):
                continue
            cur_data, cur_group, _, cur_seg = provider.loadDataFile_with_groupseglabel_stanfordindoor(cur_train_filename)

            seg_output = np.zeros_like(cur_seg)
            segrefine_output = np.zeros_like(cur_seg)
            group_output = np.zeros_like(cur_group)
            conf_output = np.zeros_like(cur_group).astype(np.float)

            pts_label_one_hot, pts_label_mask = model.convert_seg_to_one_hot(cur_seg)
            pts_group_label, _ = model.convert_groupandcate_to_one_hot(cur_group)
            num_data = cur_data.shape[0]

            gap = 5e-3
            volume_num = int(1. / gap)+1
            volume = -1* np.ones([volume_num,volume_num,volume_num]).astype(np.int32)
            volume_seg = -1* np.ones([volume_num,volume_num,volume_num, NUM_CATEGORY]).astype(np.int32)

            intersections = np.zeros(NUM_CATEGORY)
            unions = np.zeros(NUM_CATEGORY)
            print('[%d / %d] Block Number: %d' % (shape_idx, len_pts_files, num_data))
            print('Loading train file %s' % (cur_train_filename))

            flag = True
            for j in range(num_data):

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

                try:
                    groupids_block, refineseg, group_seg = GroupMerging_old(pts_corr_val, pred_confidence_val, ptsclassification_val, label_bin)  # yolo_to_groupt(pts_corr_val, pts_corr_label_val0[0], seg,t=5)
                    groupids = BlockMerging(volume, volume_seg, pts[:,6:], groupids_block.astype(np.int32), group_seg, gap)


                seg_output[j,:] = ptsclassification_val
                segrefine_output[j,:] = refineseg
                group_output[j,:] = groupids
                conf_output[j,:] = pred_confidence_val

            ###### Generate Results for Evaluation

            basefilename = os.path.basename(cur_train_filename).split('.')[-2]
            scene_fn = os.path.join(OUTPUT_DIR, '%s.txt' % basefilename)
            f_scene = open(scene_fn, 'w')
            scene_gt_fn = os.path.join(GT_DIR, '%s.txt' % basefilename)
            group_pred = group_output.reshape(-1)
            seg_pred = seg_output.reshape(-1)
            conf = conf_output.reshape(-1)
            pts = cur_data.reshape([-1, 9])

            # filtering
            x = (pts[:, 6] / gap).astype(np.int32)
            y = (pts[:, 7] / gap).astype(np.int32)
            z = (pts[:, 8] / gap).astype(np.int32)
            for i in range(group_pred.shape[0]):
                if volume[x[i], y[i], z[i]] != -1:
                    group_pred[i] = volume[x[i], y[i], z[i]]

            un = np.unique(group_pred)
            pts_in_pred = [[] for itmp in range(NUM_CATEGORY)]
            group_pred_final = -1 * np.ones_like(group_pred)
            grouppred_cnt = 0

            for ig, g in enumerate(un): #each object in prediction
                if g == -1:
                    continue
                obj_fn = "predicted_masks/%s_%d.txt" % (basefilename, ig)
                tmp = (group_pred == g)
                sem_seg_g = int(stats.mode(seg_pred[tmp])[0])
                if np.sum(tmp) > 0.25 * min_num_pts_in_group[sem_seg_g]:
                    pts_in_pred[sem_seg_g] += [tmp]
                    group_pred_final[tmp] = grouppred_cnt
                    conf_obj = np.mean(conf[tmp])
                    grouppred_cnt += 1
                    f_scene.write("%s %d %f\n" % (obj_fn, sem_seg_g, conf_obj))
                    np.savetxt(os.path.join(OUTPUT_DIR, obj_fn), tmp.astype(np.int), fmt='%d')

            seg_gt = cur_seg.reshape(-1)
            group_gt = cur_group.reshape(-1)
            groupid_gt = seg_gt * 1000 + group_gt
            np.savetxt(scene_gt_fn, groupid_gt.astype(np.int64), fmt='%d')

            f_scene.close()

            if output_verbose:
                output_color_point_cloud(pts[:, 6:], seg_pred.astype(np.int32),
                                         os.path.join(OUTPUT_DIR, '%s_segpred.obj' % (obj_fn)))
                output_color_point_cloud(pts[:, 6:], group_pred_final.astype(np.int32),
                                         os.path.join(OUTPUT_DIR, '%s_grouppred.obj' % (obj_fn)))


with tf.Graph().as_default():
    predict()