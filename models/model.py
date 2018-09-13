import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
import tensorflow as tf
import numpy as np
import tf_util
import pointnet


NUM_CATEGORY = 13
NUM_GROUPS = 50

def placeholder_inputs(batch_size, num_point, num_group, num_cate):

    if num_point == 0:
        pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, None, 9))
    else:
        pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))

    pts_seglabels_ph = tf.placeholder(tf.int32, shape=(batch_size, num_point, num_cate))
    pts_grouplabels_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_group))
    pts_seglabel_mask_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    pts_group_mask_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point))

    alpha_ph = tf.placeholder(tf.float32, shape=())

    return pointclouds_ph, pts_seglabels_ph, pts_grouplabels_ph, pts_seglabel_mask_ph, pts_group_mask_ph, alpha_ph

def convert_seg_to_one_hot(labels):
    # labels:BxN

    label_one_hot = np.zeros((labels.shape[0], labels.shape[1], NUM_CATEGORY))
    pts_label_mask = np.zeros((labels.shape[0], labels.shape[1]))

    un, cnt = np.unique(labels, return_counts=True)
    label_count_dictionary = dict(zip(un, cnt))
    totalnum = 0
    for k_un, v_cnt in label_count_dictionary.iteritems():
        if k_un != -1:
            totalnum += v_cnt

    for idx in range(labels.shape[0]):
        for jdx in range(labels.shape[1]):
            if labels[idx, jdx] != -1:
                label_one_hot[idx, jdx, labels[idx, jdx]] = 1
                pts_label_mask[idx, jdx] = float(totalnum) / float(label_count_dictionary[labels[idx, jdx]]) # 1. - float(label_count_dictionary[labels[idx, jdx]]) / totalnum

    return label_one_hot, pts_label_mask


def convert_groupandcate_to_one_hot(grouplabels):
    # grouplabels: BxN

    group_one_hot = np.zeros((grouplabels.shape[0], grouplabels.shape[1], NUM_GROUPS))
    pts_group_mask = np.zeros((grouplabels.shape[0], grouplabels.shape[1]))

    un, cnt = np.unique(grouplabels, return_counts=True)
    group_count_dictionary = dict(zip(un, cnt))
    totalnum = 0
    for k_un, v_cnt in group_count_dictionary.iteritems():
        if k_un != -1:
            totalnum += v_cnt

    for idx in range(grouplabels.shape[0]):
        un = np.unique(grouplabels[idx])
        grouplabel_dictionary = dict(zip(un, range(len(un))))
        for jdx in range(grouplabels.shape[1]):
            if grouplabels[idx, jdx] != -1:
                group_one_hot[idx, jdx, grouplabel_dictionary[grouplabels[idx, jdx]]] = 1
                pts_group_mask[idx, jdx] = float(totalnum) / float(group_count_dictionary[grouplabels[idx, jdx]]) # 1. - float(group_count_dictionary[grouplabels[idx, jdx]]) / totalnum

    return group_one_hot.astype(np.float32), pts_group_mask



def generate_group_mask(pts, grouplabels, labels):
    # grouplabels: BxN
    # pts: BxNx6
    # labels: BxN

    group_mask = np.zeros((grouplabels.shape[0], grouplabels.shape[1], grouplabels.shape[1]))

    for idx in range(grouplabels.shape[0]):
        for jdx in range(grouplabels.shape[1]):
            for kdx in range(grouplabels.shape[1]):
                if (labels[idx, jdx] == labels[idx, kdx]):
                    group_mask[idx, jdx, kdx] = 2.

                if np.linalg.norm((pts[idx, jdx, :3] - pts[idx, kdx, :3]) * (
                    pts[idx, jdx, :3] - pts[idx, kdx, :3])) < 0.04:
                    if (labels[idx, jdx] == labels[idx, kdx]):
                        group_mask[idx, jdx, kdx] = 5.
                    else:
                        group_mask[idx, jdx, kdx] = 2.

    return group_mask



def get_model(point_cloud, is_training, group_cate_num=50, m=10., bn_decay=None):
    #input: point_cloud: BxNx9 (XYZ, RGB, NormalizedXYZ)

    batch_size = point_cloud.get_shape()[0].value
    print(point_cloud.get_shape())

    F = pointnet.get_model(point_cloud, is_training, bn=True, bn_decay=bn_decay)

    # Semantic prediction
    Fsem = tf_util.conv2d(F, 128, [1, 1], padding='VALID', stride=[1, 1], bn=False, is_training=is_training, scope='Fsem')

    ptssemseg_logits = tf_util.conv2d(Fsem, group_cate_num, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None, scope='ptssemseg_logits')
    ptssemseg_logits = tf.squeeze(ptssemseg_logits, [2])

    ptssemseg = tf.nn.softmax(ptssemseg_logits, name="ptssemseg")

    # Similarity matrix
    Fsim = tf_util.conv2d(F, 128, [1, 1], padding='VALID', stride=[1, 1], bn=False, is_training=is_training, scope='Fsim')

    Fsim = tf.squeeze(Fsim, [2])

    r = tf.reduce_sum(Fsim * Fsim, 2)
    r = tf.reshape(r, [batch_size, -1, 1])
    print(r.get_shape(),Fsim.get_shape())
    D = r - 2 * tf.matmul(Fsim, tf.transpose(Fsim, perm=[0, 2, 1])) + tf.transpose(r, perm=[0, 2, 1])

    # simmat_logits = tf.maximum(D, 0.)
    simmat_logits = tf.maximum(m * D, 0.)

    # Confidence Map
    Fconf = tf_util.conv2d(F, 128, [1, 1], padding='VALID', stride=[1, 1], bn=False, is_training=is_training, scope='Fsconf')
    conf_logits = tf_util.conv2d(Fconf, 1, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None, scope='conf_logits')
    conf_logits = tf.squeeze(conf_logits, [2])

    conf = tf.nn.sigmoid(conf_logits, name="confidence")

    return {'semseg': ptssemseg,
            'semseg_logits': ptssemseg_logits,
            'simmat': simmat_logits,
            'conf': conf,
            'conf_logits': conf_logits}

def get_loss(net_output, labels, alpha=10., margin=[1.,2.]):
    """
    input:
        net_output:{'semseg', 'semseg_logits','simmat','conf','conf_logits'}
        labels:{'ptsgroup', 'semseg','semseg_mask','group_mask'}
    """

    pts_group_label = labels['ptsgroup']
    pts_semseg_label = labels['semseg']
    group_mask = tf.expand_dims(labels['group_mask'], dim=2)

    pred_confidence_logits = net_output['conf']
    pred_simmat = net_output['simmat']

    # Similarity Matrix loss
    B = pts_group_label.get_shape()[0]
    N = pts_group_label.get_shape()[1]

    onediag = tf.ones([B,N], tf.float32)

    group_mat_label = tf.matmul(pts_group_label,tf.transpose(pts_group_label, perm=[0, 2, 1])) #BxNxN: (i,j) if i and j in the same group
    group_mat_label = tf.matrix_set_diag(group_mat_label,onediag)

    sem_mat_label = tf.cast(tf.matmul(pts_semseg_label,tf.transpose(pts_semseg_label, perm=[0, 2, 1])), tf.float32) #BxNxN: (i,j) if i and j are the same semantic category
    sem_mat_label = tf.matrix_set_diag(sem_mat_label,onediag)

    samesem_mat_label = sem_mat_label
    diffsem_mat_label = tf.subtract(1.0, sem_mat_label)

    samegroup_mat_label = group_mat_label
    diffgroup_mat_label = tf.subtract(1.0, group_mat_label)
    diffgroup_samesem_mat_label = tf.multiply(diffgroup_mat_label, samesem_mat_label)
    diffgroup_diffsem_mat_label = tf.multiply(diffgroup_mat_label, diffsem_mat_label)

    num_samegroup = tf.reduce_sum(samegroup_mat_label)
    num_diffgroup_samesem = tf.reduce_sum(diffgroup_samesem_mat_label)
    num_diffgroup_diffsem = tf.reduce_sum(diffgroup_diffsem_mat_label)

    # Double hinge loss

    C_same = tf.constant(margin[0], name="C_same") # same semantic category
    C_diff = tf.constant(margin[1], name="C_diff") # different semantic category

    pos =  tf.multiply(samegroup_mat_label, pred_simmat) # minimize distances if in the same group
    neg_samesem = alpha * tf.multiply(diffgroup_samesem_mat_label, tf.maximum(tf.subtract(C_same, pred_simmat), 0))
    neg_diffsem = tf.multiply(diffgroup_diffsem_mat_label, tf.maximum(tf.subtract(C_diff, pred_simmat), 0))


    simmat_loss = neg_samesem + neg_diffsem + pos
    group_mask_weight = tf.matmul(group_mask, tf.transpose(group_mask, perm=[0, 2, 1]))
    # simmat_loss = tf.add(simmat_loss, pos)
    simmat_loss = tf.multiply(simmat_loss, group_mask_weight)

    simmat_loss = tf.reduce_mean(simmat_loss)

    # Semantic Segmentation loss
    ptsseg_loss = tf.nn.softmax_cross_entropy_with_logits(logits=net_output['semseg_logits'], labels=pts_semseg_label)
    ptsseg_loss = tf.multiply(ptsseg_loss, labels['semseg_mask'])
    ptsseg_loss = tf.reduce_mean(ptsseg_loss)

    # Confidence Map loss
    Pr_obj = tf.reduce_sum(pts_semseg_label,axis=2)
    Pr_obj = tf.cast(Pr_obj, tf.float32)
    ng_label = group_mat_label
    ng_label = tf.greater(ng_label, tf.constant(0.5))
    ng = tf.less(pred_simmat, tf.constant(margin[0]))

    epsilon = tf.constant(np.ones(ng_label.get_shape()[:2]).astype(np.float32) * 1e-6)
    pts_iou = tf.div(tf.reduce_sum(tf.cast(tf.logical_and(ng,ng_label), tf.float32), axis=2),
                     (tf.reduce_sum(tf.cast(tf.logical_or(ng,ng_label), tf.float32), axis=2)+epsilon))
    confidence_label = tf.multiply(pts_iou, Pr_obj) # BxN

    confidence_loss = tf.reduce_mean(tf.squared_difference(confidence_label, tf.squeeze(pred_confidence_logits,[2])))

    loss = simmat_loss + ptsseg_loss + confidence_loss

    grouperr = tf.abs(tf.cast(ng, tf.float32) - tf.cast(ng_label, tf.float32))

    return loss, tf.reduce_mean(grouperr), \
           tf.reduce_sum(grouperr * diffgroup_samesem_mat_label), num_diffgroup_samesem, \
           tf.reduce_sum(grouperr * diffgroup_diffsem_mat_label), num_diffgroup_diffsem, \
           tf.reduce_sum(grouperr * samegroup_mat_label), num_samegroup