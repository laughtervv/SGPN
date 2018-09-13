import numpy as np
from scipy import stats
import matplotlib as mpl
import json
mpl.use('Agg')

############################
##    Ths Statistics      ##
############################

def Get_Ths(pts_corr, seg, ins, ths, ths_, cnt):

    pts_in_ins = {}
    for ip, pt in enumerate(pts_corr):
        if ins[ip] in pts_in_ins.keys():
            pts_in_curins_ind = pts_in_ins[ins[ip]]
            pts_notin_curins_ind = (~(pts_in_ins[ins[ip]])) & (seg==seg[ip])
            hist, bin = np.histogram(pt[pts_in_curins_ind], bins=20)

            if seg[ip]==8:
                print bin

            numpt_in_curins = np.sum(pts_in_curins_ind)
            numpt_notin_curins = np.sum(pts_notin_curins_ind)

            if numpt_notin_curins > 0:

                tp_over_fp = 0
                ib_opt = -2
                for ib, b in enumerate(bin):
                    if b == 0:
                        break
                    tp = float(np.sum(pt[pts_in_curins_ind] < bin[ib])) / float(numpt_in_curins)
                    fp = float(np.sum(pt[pts_notin_curins_ind] < bin[ib])) / float(numpt_notin_curins)

                    if tp <= 0.5:
                        continue

                    if fp == 0. and tp > 0.5:
                        ib_opt = ib
                        break

                    if tp/fp > tp_over_fp:
                        tp_over_fp = tp / fp
                        ib_opt = ib

                if tp_over_fp >  4.:
                    ths[seg[ip]] += bin[ib_opt]
                    ths_[seg[ip]] += bin[ib_opt]
                    cnt[seg[ip]] += 1

        else:
            pts_in_curins_ind = (ins == ins[ip])
            pts_in_ins[ins[ip]] = pts_in_curins_ind
            pts_notin_curins_ind = (~(pts_in_ins[ins[ip]])) & (seg==seg[ip])
            hist, bin = np.histogram(pt[pts_in_curins_ind], bins=20)

            if seg[ip]==8:
                print bin

            numpt_in_curins = np.sum(pts_in_curins_ind)
            numpt_notin_curins = np.sum(pts_notin_curins_ind)

            if numpt_notin_curins > 0:

                tp_over_fp = 0
                ib_opt = -2
                for ib, b in enumerate(bin):

                    if b == 0:
                        break

                    tp = float(np.sum(pt[pts_in_curins_ind]<bin[ib])) / float(numpt_in_curins)
                    fp = float(np.sum(pt[pts_notin_curins_ind]<bin[ib])) / float(numpt_notin_curins)

                    if tp <= 0.5:
                        continue

                    if fp == 0. and tp > 0.5:
                        ib_opt = ib
                        break

                    if tp / fp > tp_over_fp:
                        tp_over_fp = tp / fp
                        ib_opt = ib

                if tp_over_fp >  4.:
                    ths[seg[ip]] += bin[ib_opt]
                    ths_[seg[ip]] += bin[ib_opt]
                    cnt[seg[ip]] += 1

    return ths, ths_, cnt


##############################
##    Merging Algorithms    ##
##############################

def GroupMerging(pts_corr, confidence, seg, label_bin):

    confvalidpts = (confidence>0.4)
    un_seg = np.unique(seg)
    refineseg = -1* np.ones(pts_corr.shape[0])
    groupid = -1* np.ones(pts_corr.shape[0])
    numgroups = 0
    groupseg = {}
    for i_seg in un_seg:
        if i_seg==-1:
            continue
        pts_in_seg = (seg==i_seg)
        valid_seg_group = np.where(pts_in_seg & confvalidpts)
        proposals = []
        if valid_seg_group[0].shape[0]==0:
            proposals += [pts_in_seg]
        else:
            for ip in valid_seg_group[0]:
                validpt = (pts_corr[ip] < label_bin[i_seg]) & pts_in_seg
                if np.sum(validpt)>5:
                    flag = False
                    for gp in range(len(proposals)):
                        iou = float(np.sum(validpt & proposals[gp])) / np.sum(validpt|proposals[gp])#uniou
                        validpt_in_gp = float(np.sum(validpt & proposals[gp])) / np.sum(validpt)#uniou
                        if iou > 0.6 or validpt_in_gp > 0.8:
                            flag = True
                            if np.sum(validpt)>np.sum(proposals[gp]):
                                proposals[gp] = validpt
                            continue

                    if not flag:
                        proposals += [validpt]

            if len(proposals) == 0:
                proposals += [pts_in_seg]
        for gp in range(len(proposals)):
            if np.sum(proposals[gp])>50:
                groupid[proposals[gp]] = numgroups
                groupseg[numgroups] = i_seg
                numgroups += 1
                refineseg[proposals[gp]] = stats.mode(seg[proposals[gp]])[0]

    un, cnt = np.unique(groupid, return_counts=True)
    for ig, g in enumerate(un):
        if cnt[ig] < 50:
            groupid[groupid==g] = -1

    un, cnt = np.unique(groupid, return_counts=True)
    groupidnew = groupid.copy()
    for ig, g in enumerate(un):
        if g == -1:
            continue
        groupidnew[groupid==g] = (ig-1)
        groupseg[(ig-1)] = groupseg.pop(g)
    groupid = groupidnew

    for ip, gid in enumerate(groupid):
        if gid == -1:
            pts_in_gp_ind = (pts_corr[ip] < label_bin[seg[ip]])
            pts_in_gp = groupid[pts_in_gp_ind]
            pts_in_gp_valid = pts_in_gp[pts_in_gp!=-1]
            if len(pts_in_gp_valid) != 0:
                groupid[ip] = stats.mode(pts_in_gp_valid)[0][0]

    return groupid, refineseg, groupseg

def BlockMerging(volume, volume_seg, pts, grouplabel, groupseg, gap=1e-3):

    overlapgroupcounts = np.zeros([100,300])
    groupcounts = np.ones(100)
    x=(pts[:,0]/gap).astype(np.int32)
    y=(pts[:,1]/gap).astype(np.int32)
    z=(pts[:,2]/gap).astype(np.int32)
    for i in range(pts.shape[0]):
        xx=x[i]
        yy=y[i]
        zz=z[i]
        if grouplabel[i] != -1:
            if volume[xx,yy,zz]!=-1 and volume_seg[xx,yy,zz]==groupseg[grouplabel[i]]:
                overlapgroupcounts[grouplabel[i],volume[xx,yy,zz]] += 1
        groupcounts[grouplabel[i]] += 1

    groupcate = np.argmax(overlapgroupcounts,axis=1)
    maxoverlapgroupcounts = np.max(overlapgroupcounts,axis=1)

    curr_max = np.max(volume)
    for i in range(groupcate.shape[0]):
        if maxoverlapgroupcounts[i]<7 and groupcounts[i]>30:
            curr_max += 1
            groupcate[i] = curr_max


    finalgrouplabel = -1 * np.ones(pts.shape[0])

    for i in range(pts.shape[0]):
        if grouplabel[i] != -1 and volume[x[i],y[i],z[i]]==-1:
            volume[x[i],y[i],z[i]] = groupcate[grouplabel[i]]
            volume_seg[x[i],y[i],z[i]] = groupseg[grouplabel[i]]
            finalgrouplabel[i] = groupcate[grouplabel[i]]
    return finalgrouplabel


############################
##    Evaluation Metrics  ##
############################

def eval_3d_perclass(tp, fp, npos):

    tp = np.asarray(tp).astype(np.float)
    fp = np.asarray(fp).astype(np.float)
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    rec = tp / npos
    prec = tp / (fp+tp)

    ap = 0.
    for t in np.arange(0, 1, 0.1):
        prec1 = prec[rec>=t]
        prec1 = prec1[~np.isnan(prec1)]
        if len(prec1) == 0:
            p = 0.
        else:
            p = max(prec1)
            if not p:
                p = 0.

        ap = ap + p / 10


    return ap, rec, prec

############################
##    Visualize Results   ##
############################

color_map = json.load(open('part_color_mapping.json', 'r'))

def output_bounding_box_withcorners(box_corners, seg, out_file):
    # ##############   0       1       2       3       4       5       6       7
    corner_indexes = [[0, 1, 2], [0, 1, 5], [0, 4, 2], [0, 4, 5], [3, 1, 2], [3, 1, 5], [3, 4, 2], [3, 4, 5]]
    line_indexes = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    with open(out_file, 'w') as f:
        l = box_corners.shape[0]
        for i in range(l):
            box = box_corners[i]
            color = color_map[seg[i]]
            for line_index in line_indexes:
                corner0 = box[line_index[0]]
                corner1 = box[line_index[1]]
                print corner0.shape
                dist = np.linalg.norm(corner0 - corner1)
                dot_num = int(dist / 0.005)
                delta = (corner1 - corner0) / dot_num
                for idot in range(dot_num):
                    plotdot = corner0 + idot * delta
                    f.write(
                        'v %f %f %f %f %f %f\n' % (plotdot[0], plotdot[1], plotdot[2], color[0], color[1], color[2]))


def output_bounding_box(boxes, seg, out_file):
    # ##############   0       1       2       3       4       5       6       7
    #box:nx8x3
    corner_indexes = [[0, 1, 2], [0, 1, 5], [0, 4, 2], [0, 4, 5], [3, 1, 2], [3, 1, 5], [3, 4, 2], [3, 4, 5]]
    line_indexes = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    with open(out_file, 'w') as f:
        l = boxes.shape[0]
        for i in range(l):
            box = boxes[i]
            color = color_map[seg[i]]
            for line_index in line_indexes:
                corner0 = box[corner_indexes[line_index[0]]]
                corner1 = box[corner_indexes[line_index[1]]]
                dist = np.linalg.norm(corner0 - corner1)
                dot_num = int(dist / 0.005)
                delta = (corner1 - corner0) / dot_num
                for idot in range(dot_num):
                    plotdot = corner0 + idot * delta
                    f.write(
                        'v %f %f %f %f %f %f\n' % (plotdot[0], plotdot[1], plotdot[2], color[0], color[1], color[2]))


def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def output_point_cloud_rgb(data, rgb, out_file):
    with open(out_file, 'w') as f:
        l = len(data)
        for i in range(l):
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], rgb[i][0],  rgb[i][1],  rgb[i][2]))


def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


##define color heat map
norm = mpl.colors.Normalize(vmin=0, vmax=255)
magma_cmap = mpl.cm.get_cmap('magma')
magma_rgb = []
for i in range(0, 255):
       k = mpl.colors.colorConverter.to_rgb(magma_cmap(norm(i)))
       magma_rgb.append(k)


def output_scale_point_cloud(data, scales, out_file):
    with open(out_file, 'w') as f:
        l = len(scales)
        for i in range(l):
            scale = int(scales[i]*254)
            if scale > 254:
                scale = 254
            color = magma_rgb[scale]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

