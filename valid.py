import argparse
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import provider
from utils.test_utils import *
from models import model

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="1", help='GPU to use [default: GPU 1]')
parser.add_argument('--verbose', action='store_true', help='if specified, use depthconv')
parser.add_argument('--input_list', type=str, default='/media/hdd2/data/pointnet/stanfordindoor/valid_hdf5_file_list.txt', help='Validation data list')
parser.add_argument('--restore_dir', type=str, default='checkpoint/stanford_ins_seg_groupmask11_fromgroup_recipweight_nopow2_lr4', help='Directory that stores all training logs and trained models')
FLAGS = parser.parse_args()

PRETRAINED_MODEL_PATH = os.path.join(FLAGS.restore_dir,'trained_models/')

gpu_to_use = 0
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

RESTORE_DIR = FLAGS.restore_dir
OUTPUT_DIR = os.path.join(FLAGS.restore_dir, 'valid_results')
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

OUTPUT_VERBOSE = FLAGS.verbose  # If true, output similarity

# MAIN SCRIPT
POINT_NUM = 4096  # the max number of points in the all testing data shapes
BATCH_SIZE = 1
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

        group_mat_label = tf.matmul(ptsgroup_label_ph, tf.transpose(ptsgroup_label_ph, perm=[0, 2, 1]))
        net_output = model.get_model(pointclouds_ph, is_training_ph, group_cate_num=NUM_CATEGORY)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:

        # Restore variables from disk.

        ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH,os.path.basename(ckptstate.model_checkpoint_path))
            saver.restore(sess, LOAD_MODEL_FILE)
            print ("Model loaded in file: %s" % LOAD_MODEL_FILE)
        else:
            print ("Fail to load modelfile: %s" % PRETRAINED_MODEL_PATH)

        ths = np.zeros(NUM_CATEGORY)
        ths_ = np.zeros(NUM_CATEGORY)
        cnt = np.zeros(NUM_CATEGORY)
        min_groupsize = np.zeros(NUM_CATEGORY)
        min_groupsize_cnt = np.zeros(NUM_CATEGORY)


        for shape_idx in range(len_pts_files):

            cur_train_filename = test_file_list[shape_idx]

            if not os.path.exists(cur_train_filename):
                continue
            cur_data, cur_group, _, cur_seg = provider.loadDataFile_with_groupseglabel_stanfordindoor(cur_train_filename)

            if OUTPUT_VERBOSE:
                pts = np.reshape(cur_data, [-1,9])
                output_point_cloud_rgb(pts[:, 6:], pts[:, 3:6], os.path.join(OUTPUT_DIR, '%d_pts.obj' % (shape_idx)))

            pts_label_one_hot, pts_label_mask = model.convert_seg_to_one_hot(cur_seg)
            pts_group_label, _ = model.convert_groupandcate_to_one_hot(cur_group)
            num_data = cur_data.shape[0]

            cur_seg_flatten = np.reshape(cur_seg, [-1])
            un, indices = np.unique(cur_group, return_index=True)
            for iu, u in enumerate(un):
                groupsize = np.sum(cur_group == u)
                groupcate = cur_seg_flatten[indices[iu]]
                min_groupsize[groupcate] += groupsize
                # print groupsize, min_groupsize[groupcate]/min_groupsize_cnt[groupcate]
                min_groupsize_cnt[groupcate] += 1

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

                pts_corr_val = np.squeeze(pts_corr_val0[0])
                pred_confidence_val = np.squeeze(pred_confidence_val0[0])
                ptsclassification_val = np.argmax(np.squeeze(ptsclassification_val0[0]),axis=1)

                pts_corr_label_val = np.squeeze(1 - pts_corr_label_val0)
                seg = np.squeeze(seg)
                ins = np.squeeze(ins)

                ind = (seg == 8)
                pts_corr_val0 = (pts_corr_val > 1.).astype(np.float)
                print np.mean(np.transpose(np.abs(pts_corr_label_val[ind] - pts_corr_val0[ind]),axes=[1,0])[ind])

                ths, ths_, cnt = Get_Ths(pts_corr_val, seg, ins, ths, ths_, cnt)
                print ths/cnt


                if OUTPUT_VERBOSE:
                    un,indices = np.unique(ins,return_index=True)
                    for ii,id in enumerate(indices):
                        corr = pts_corr_val[id].copy()
                        output_scale_point_cloud(pts[:,6:], np.float32(corr), os.path.join(OUTPUT_DIR, '%d_%d_%d_%d_scale.obj'%(shape_idx,j,un[ii],seg[id])))
                        corr = pts_corr_label_val[id]
                        output_scale_point_cloud(pts[:, 6:], np.float32(corr), os.path.join(OUTPUT_DIR, '%d_%d_%d_%d_scalegt.obj' % (shape_idx, j, un[ii],seg[id])))
                    output_scale_point_cloud(pts[:, 6:], np.float32(pred_confidence_val), os.path.join(OUTPUT_DIR, '%d_%d_conf.obj' % (shape_idx, j)))
                    output_color_point_cloud(pts[:,6:], ptsclassification_val.astype(np.int32), os.path.join(OUTPUT_DIR, '%d_seg.obj'%(shape_idx)))

        ths = [ths[i]/cnt[i] if cnt[i] != 0 else 0.2 for i in range(len(cnt))]
        np.savetxt(os.path.join(RESTORE_DIR, 'pergroup_thres.txt'), ths)

        min_groupsize = [int(float(min_groupsize[i]) / min_groupsize_cnt[i]) if min_groupsize_cnt[i] != 0 else 0 for i in range(len(min_groupsize))]
        np.savetxt(os.path.join(RESTORE_DIR, 'mingroupsize.txt'), min_groupsize)

with tf.Graph().as_default():
    predict()
