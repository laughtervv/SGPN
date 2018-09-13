import argparse
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../../'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../models'))
import provider
from models import model

# Parsing Arguments
parser = argparse.ArgumentParser()
# Experiment Settings
parser.add_argument('--gpu', type=str, default="1", help='GPU to use [default: GPU 1]')
parser.add_argument('--wd', type=float, default=0.9, help='Weight Decay [Default: 0.0]')
parser.add_argument('--epoch', type=int, default=200, help='Number of epochs [default: 50]')
parser.add_argument('--batch', type=int, default=4, help='Batch Size during training [default: 4]')
parser.add_argument('--point_num', type=int, default=4096, help='Point Number')
parser.add_argument('--group_num', type=int, default=50, help='Maximum Group Number in one pc')
parser.add_argument('--cate_num', type=int, default=13, help='Number of categories')
parser.add_argument('--margin_same', type=float, default=10., help='Double hinge loss margin: same semantic')
parser.add_argument('--margin_diff', type=float, default=80., help='Double hinge loss margin: different semantic')

# Input&Output Settings
parser.add_argument('--output_dir', type=str, default='checkpoint/stanford_sem_seg', help='Directory that stores all training logs and trained models')
parser.add_argument('--input_list', type=str, default='data/train_hdf5_file_list.txt', help='Input data list file')
parser.add_argument('--restore_model', type=str, default='checkpoint/stanford_ins_seg', help='Pretrained model')

FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

TRAINING_FILE_LIST = FLAGS.input_list
PRETRAINED_MODEL_PATH = os.path.join(FLAGS.restore_model, 'trained_models/')

POINT_NUM = FLAGS.point_num
BATCH_SIZE = FLAGS.batch
OUTPUT_DIR = FLAGS.output_dir

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

NUM_GROUPS = FLAGS.group_num
NUM_CATEGORY = FLAGS.cate_num

print('#### Batch Size: {0}'.format(BATCH_SIZE))
print('#### Point Number: {0}'.format(POINT_NUM))
print('#### Training using GPU: {0}'.format(FLAGS.gpu))

DECAY_STEP = 800000.
DECAY_RATE = 0.5

LEARNING_RATE_CLIP = 1e-6
BASE_LEARNING_RATE = 1e-4
MOMENTUM = 0.9

TRAINING_EPOCHES = FLAGS.epoch
MARGINS = [FLAGS.margin_same, FLAGS.margin_diff]

print('### Training epoch: {0}'.format(TRAINING_EPOCHES))

MODEL_STORAGE_PATH = os.path.join(OUTPUT_DIR, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

LOG_STORAGE_PATH = os.path.join(OUTPUT_DIR, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
    os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER = os.path.join(OUTPUT_DIR, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)

LOG_DIR = FLAGS.output_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

os.system('cp %s %s' % (os.path.join(BASE_DIR, 'models/model.py'), LOG_DIR))  # bkp of model def
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train.py'), LOG_DIR))  # bkp of train procedure

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(FLAGS.gpu)):
            batch = tf.Variable(0, trainable=False, name='batch')
            learning_rate = tf.train.exponential_decay(
                BASE_LEARNING_RATE,  # base learning rate
                batch * BATCH_SIZE,  # global_var indicating the number of steps
                DECAY_STEP,  # step size
                DECAY_RATE,  # decay rate
                staircase=True  # Stair-case or continuous decreasing
            )
            learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP)

            lr_op = tf.summary.scalar('learning_rate', learning_rate)

            pointclouds_ph, ptsseglabel_ph, ptsgroup_label_ph, pts_seglabel_mask_ph, pts_group_mask_ph, alpha_ph = \
                model.placeholder_inputs(BATCH_SIZE, POINT_NUM, NUM_GROUPS, NUM_CATEGORY)
            is_training_ph = tf.placeholder(tf.bool, shape=())

            labels = {'ptsgroup': ptsgroup_label_ph,
                      'semseg': ptsseglabel_ph,
                      'semseg_mask': pts_seglabel_mask_ph,
                      'group_mask': pts_group_mask_ph}

            net_output = model.get_model(pointclouds_ph, is_training_ph, group_cate_num=NUM_CATEGORY, m=MARGINS[0])
            loss, grouperr, same, same_cnt, diff, diff_cnt, pos, pos_cnt = model.get_loss(net_output, labels, alpha_ph, MARGINS)

            total_training_loss_ph = tf.placeholder(tf.float32, shape=())
            group_err_loss_ph = tf.placeholder(tf.float32, shape=())
            total_train_loss_sum_op = tf.summary.scalar('total_training_loss', total_training_loss_ph)
            group_err_op = tf.summary.scalar('group_err_loss', group_err_loss_ph)

        train_variables = tf.trainable_variables()

        trainer = tf.train.AdamOptimizer(learning_rate)
        train_op = trainer.minimize(loss, var_list=train_variables, global_step=batch)

        loader = tf.train.Saver([v for v in tf.all_variables()#])
                                 if
                                   ('conf_logits' not in v.name) and
                                    ('Fsim' not in v.name) and
                                    ('Fsconf' not in v.name) and
                                    ('batch' not in v.name)
                                ])

        saver = tf.train.Saver([v for v in tf.all_variables()])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        init = tf.global_variables_initializer()
        sess.run(init)

        train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)

        train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
        num_train_file = len(train_file_list)

        fcmd = open(os.path.join(LOG_STORAGE_PATH, 'cmd.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()

        flog = open(os.path.join(LOG_STORAGE_PATH, 'log.txt'), 'w')

        ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
            loader.restore(sess, LOAD_MODEL_FILE)
            printout(flog, "Model loaded in file: %s" % LOAD_MODEL_FILE)
        else:
            printout(flog, "Fail to load modelfile: %s" % PRETRAINED_MODEL_PATH)


        train_file_idx = np.arange(0, len(train_file_list))
        np.random.shuffle(train_file_idx)

        ## load all data into memory
        all_data = []
        all_group = []
        all_seg = []
        for i in range(num_train_file):
            cur_train_filename = train_file_list[train_file_idx[i]]
            # printout(flog, 'Loading train file ' + cur_train_filename)
            cur_data, cur_group, _, cur_seg = provider.loadDataFile_with_groupseglabel_stanfordindoor(cur_train_filename)
            all_data += [cur_data]
            all_group += [cur_group]
            all_seg += [cur_seg]

        all_data = np.concatenate(all_data,axis=0)
        all_group = np.concatenate(all_group,axis=0)
        all_seg = np.concatenate(all_seg,axis=0)

        num_data = all_data.shape[0]
        num_batch = num_data // BATCH_SIZE

        def train_one_epoch(epoch_num):

            ### NOTE: is_training = False: We do not update bn parameters during training due to the small batch size. This requires pre-training PointNet with large batchsize (say 32).
            is_training = False

            order = np.arange(num_data)
            np.random.shuffle(order)

            total_loss = 0.0
            total_grouperr = 0.0
            total_same = 0.0
            total_diff = 0.0
            total_pos = 0.0
            same_cnt0 = 0

            for j in range(num_batch):
                begidx = j * BATCH_SIZE
                endidx = (j + 1) * BATCH_SIZE

                pts_label_one_hot, pts_label_mask = model.convert_seg_to_one_hot(all_seg[order[begidx: endidx]])
                pts_group_label, pts_group_mask = model.convert_groupandcate_to_one_hot(all_group[order[begidx: endidx]])

                feed_dict = {
                    pointclouds_ph: all_data[order[begidx: endidx], ...],
                    ptsseglabel_ph: pts_label_one_hot,
                    ptsgroup_label_ph: pts_group_label,
                    pts_seglabel_mask_ph: pts_label_mask,
                    pts_group_mask_ph: pts_group_mask,
                    is_training_ph: is_training,
                    alpha_ph: min(10., (float(epoch_num) / 5.) * 2. + 2.),
                }

                _, loss_val, simmat_val, grouperr_val, same_val, same_cnt_val, diff_val, diff_cnt_val, pos_val, pos_cnt_val = sess.run([train_op, loss, net_output['simmat'], grouperr, same, same_cnt, diff, diff_cnt, pos, pos_cnt], feed_dict=feed_dict)
                total_loss += loss_val
                total_grouperr += grouperr_val
                total_diff += (diff_val / diff_cnt_val)
                if same_cnt_val > 0:
                    total_same += same_val / same_cnt_val
                    same_cnt0 += 1
                total_pos += pos_val / pos_cnt_val


                if j % 10 == 9:
                    printout(flog, 'Batch: %d, loss: %f, grouperr: %f, same: %f, diff: %f, pos: %f' % (j, total_loss/10, total_grouperr/10, total_same/same_cnt0, total_diff/10, total_pos/10))

                    lr_sum, batch_sum, train_loss_sum, group_err_sum = sess.run( \
                        [lr_op, batch, total_train_loss_sum_op, group_err_op], \
                        feed_dict={total_training_loss_ph: total_loss / 10.,
                                   group_err_loss_ph: total_grouperr / 10., })

                    train_writer.add_summary(train_loss_sum, batch_sum)
                    train_writer.add_summary(lr_sum, batch_sum)
                    train_writer.add_summary(group_err_sum, batch_sum)

                    total_grouperr = 0.0
                    total_loss = 0.0
                    total_diff = 0.0
                    total_same = 0.0
                    total_pos = 0.0
                    same_cnt0 = 0



            cp_filename = saver.save(sess,
                                     os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch_num + 1) + '.ckpt'))
            printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)

        if not os.path.exists(MODEL_STORAGE_PATH):
            os.mkdir(MODEL_STORAGE_PATH)

        for epoch in range(TRAINING_EPOCHES):
            printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))

            train_file_idx = np.arange(0, len(train_file_list))
            np.random.shuffle(train_file_idx)

            train_one_epoch(epoch)
            flog.flush()

            cp_filename = saver.save(sess,
                                     os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch + 1) + '.ckpt'))
            printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)


        flog.close()


if __name__ == '__main__':
    train()
