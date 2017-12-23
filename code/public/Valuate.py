#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:   construct the evaluation program

from __future__ import print_function
import time
import pandas as pd
import numpy as np
import math
import datetime
from numpy import maximum
from numpy import greater
from pandas import DataFrame
import os
__docformat__ = 'restructedtext en'


def exe_time(func):
    def new_func(*args, **args2):
        t0 = time.time()
        print("-- @%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("-- @%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("-- @%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return new_func


def fun_hit_zero_one(user_test_recom):
    """
    根据recom_list中item在test_lst里的出现情况生成与recom_list等长的0/1序列
    0表示推荐的item不在test里，1表示推荐的item在test里
    :param test_lst: 单个用户的test列表
    :param recom_lst: 推荐的列表
    :param test_mask: 单个用户的test列表对应的mask列表
    :return: 与recom_list等长的0/1序列。
    """
    test_lst, recom_lst, test_mask, _ = user_test_recom
    test_lst = test_lst[:np.sum(test_mask)]     # 取出来有效的user_test_list
    seq = []
    for e in recom_lst:
        if e in test_lst:       # 命中
            seq.append(1)
        else:                   # 没命中
            seq.append(0)
    return np.array(seq)


def fun_hit_recall_item_idx(user_test_recom):
    """
    根据recom_list中item在test_lst里的出现情况生成与recom_list等长的-1/idx序列
    -1表示推荐的item不在test里，idx表示推荐的item在test里所对应的item_index
    :param test_lst: 单个用户的test列表
    :param recom_lst: 推荐的列表
    :param test_mask: 单个用户的test列表对应的mask列表
    :return: 与recom_list等长的-1/idx序列。
    """
    test_lst, recom_lst, test_mask, _ = user_test_recom
    test_lst = test_lst[:np.sum(test_mask)]     # 取出来有效的user_test_list
    idxs = []
    for e in recom_lst:
        if e in test_lst:       # 命中
            idxs.append(e)
        else:                   # 没命中。不能标为0，因为有一个item的别名是0
            idxs.append(-1)     # 也不要标为np.nan，会有奇葩的错误
    return np.array(idxs)


def fun_hit_auc_item_idx(test_upqs):
    """
    算auc时，已得出所有用户的所有test里正负计算，正的为true, 负的为false，得到那些为true的items_id
    -1表示推荐的item不在test里，idx表示推荐的item在test里所对应的item_index
    :param test_lst: 单个用户的test列表
    :param upq_lst: 正负计算得出来的正负列表
    :param test_mask: 单个用户的test列表对应的mask列表
    :return: 与upq_lst等长的标号为1对应的item_idx序列
    """
    test_lst, upq_lst, test_mask, _ = test_upqs
    idxs = []
    for i, e in enumerate(test_lst):
        if 1 == test_mask[i]:       # 这是前边那块有效的test
            if upq_lst[i]:
                idxs.append(e)
            else:
                idxs.append(-1)
        else:                       # test后边补0的位置。
            idxs.append(-1)
    return np.array(idxs)


def fun_item_idx_to_intervals(item_idxs, test_i_cou, p_inters):
    """
    根据标号矩阵找出标号为1的item_id，根据其在test里的出现次数，计算分别在各个区间里出现了多少交易量
    :param item_idxs: 命中的item_index
    :param test_i_cou: test里items出现次数的dict
    :param p_inters: [2, 10, 30]
    :return: 每个区间里各有多少交易量
    """
    intervals = np.array([0 for _ in range(p_inters[1])])     # 有这么些个区间
    hit_item_idxs = [idx for u_idxs in item_idxs
                     for idx in u_idxs if -1 != idx]        # 先取出来那些item_id
    for idx in hit_item_idxs:                                       # 把item_id分到各个区间
        inter = int(math.ceil(1.0 * test_i_cou[idx] / p_inters[0])) - 1
        if inter >= p_inters[1] - 1:
            inter = -1
        intervals[inter] += 1
    return intervals


def fun_evaluate_map(user_test_recom_zero_one):
    """
    计算map。所得是单个用户test的，最后所有用户的求和取平均
    :param test_lst: 单个用户的test集
    :param zero_one: 0/1序列
    :param test_mask: 单个用户的test列表对应的mask列表
    :return:
    """
    test_lst, zero_one, test_mask, _ = user_test_recom_zero_one
    test_lst = test_lst[:np.sum(test_mask)]

    zero_one = np.array(zero_one)
    if 0 == sum(zero_one):    # 没有命中的
        return 0.0
    zero_one_cum = zero_one.cumsum()                # precision要算累计命中
    zero_one_cum *= zero_one                        # 取出命中为1的那些，其余位置得0
    idxs = list(np.nonzero(zero_one_cum))[0]        # 得到(n,)类型的非零的索引array
    s = 0.0
    for idx in idxs:
        s += 1.0 * zero_one_cum[idx] / (idx + 1)
    return s / len(test_lst)


def fun_evaluate_ndcg(user_test_recom_zero_one):
    """
    计算ndcg。所得是单个用户test的，最后所有用户的求和取平均
    :param test_lst: 单个用户的test集
    :param zero_one: 0/1序列
    :param test_mask: 单个用户的test列表对应的mask列表
    :return:
    """
    test_lst, zero_one, test_mask, _ = user_test_recom_zero_one
    test_lst = test_lst[:np.sum(test_mask)]

    zero_one = np.array(zero_one)
    if 0 == sum(zero_one):    # 没有命中的
        return 0.0
    s = 0.0
    idxs = list(np.nonzero(zero_one))[0]
    for idx in idxs:
        s += 1.0 / np.log2(idx + 2)
    m = 0.0
    length = min(len(test_lst), len(zero_one))      # 序列短的，都命中为1，此时是最优情况
    for idx in range(length):
        m += 1.0 / np.log2(idx + 2)
    return s / m


def fun_idxs_of_max_n_score(user_scores_to_all_items, top_k):
    # 从一个向量里找到前n个大数所对应的index
    return np.argpartition(user_scores_to_all_items, -top_k)[-top_k:]


def fun_sort_idxs_max_to_min(user_max_n_idxs_scores):
    # 按照前n个大数index对应的具体值由大到小排序，即最左侧的index对应原得分值的最大实数值
    # 就是生成的推荐items列表里，最左侧的是得分最高的
    idxs, scores = user_max_n_idxs_scores           # idxs是n个得分最大的items，scores是所有items的得分。
    return idxs[np.argsort(scores[idxs])][::-1]     # idxs按照对应得分由大到小排列


def fun_predict_auc_recall_map_ndcg(
        p, model, best, epoch, starts_ends_auc, starts_ends_tes,
        tes_buys_masks, tes_masks,
        test_i_cou, test_i_intervals_cumsum, test_i_cold_active):
    # ------------------------------------------------------------------------------------------------------------------
    # 注意：当zip里的所有子项tes_buys_masks, all_upqs维度一致时，也就是子项的每行每列长度都一样。
    #      zip后的arr会变成三维矩阵，扫描axis=1会出错得到的一行实际上是2d array，所以后边单独加一列 append 避免该问题。
    append = [[0] for _ in np.arange(len(tes_buys_masks))]

    # ------------------------------------------------------------------------------------------------------------------
    # auc
    all_upqs = np.array([[0 for _ in np.arange(len(tes_masks[0]))]])    # 初始化
    for start_end in starts_ends_auc:
        sub_all_upqs = model.compute_sub_auc_preference(start_end)
        all_upqs = np.concatenate((all_upqs, sub_all_upqs))
    all_upqs = np.delete(all_upqs, 0, axis=0)
    # 计算：auc_intervals的当前epoch值。
    auc = 1.0 * np.sum(all_upqs) / np.sum(tes_masks)    # 全部items
    auc_item_idxs = np.apply_along_axis(    # 找出all_upqs里标为true的那些items的标号
        func1d=fun_hit_auc_item_idx,
        axis=1,
        arr=np.array(zip(tes_buys_masks, all_upqs, tes_masks, append)))
    auc_intervals = fun_item_idx_to_intervals(
        auc_item_idxs,
        test_i_cou,
        p['intervals'])
    # 保存：保存auc的最佳值
    auc_cold_active = np.array([sum(auc_intervals[:2]), sum(auc_intervals[2:])])
    auc_intervals_cumsum = auc_intervals.cumsum()
    if auc > best.best_auc:
        best.best_auc = auc
        best.best_epoch_auc = epoch
        best.best_auc_cold_active = 1.0 * auc_cold_active / test_i_cold_active
        best.best_auc_intervals_cumsum = 1.0 * auc_intervals_cumsum / test_i_intervals_cumsum

    # ------------------------------------------------------------------------------------------------------------------
    # recall, map, ndcg
    at_nums = p['at_nums']          # [5, 10, 15, 20, 30, 50]
    ranges = range(len(at_nums))

    # 计算：所有用户对所有商品预测得分的前50个。
    # 不会预测出来添加的那个虚拟商品，因为先把它从item表达里去掉
    # 注意矩阵形式的索引 all_scores[0, rank]：表示all_scores的各行里取的各个列值是rank里的各项
    all_ranks = np.array([[0 for _ in np.arange(at_nums[-1])]])   # 初始shape=(1, 50)
    for start_end in starts_ends_tes:
        sub_all_scores = model.compute_sub_all_scores(start_end)  # shape=(sub_n_user, n_item)
        sub_score_ranks = np.apply_along_axis(
            func1d=fun_idxs_of_max_n_score,
            axis=1,
            arr=sub_all_scores,
            top_k=at_nums[-1])
        sub_all_ranks = np.apply_along_axis(
            func1d=fun_sort_idxs_max_to_min,
            axis=1,
            arr=np.array(zip(sub_score_ranks, sub_all_scores)))
        all_ranks = np.concatenate((all_ranks, sub_all_ranks))
        del sub_all_scores
    all_ranks = np.delete(all_ranks, 0, axis=0)     # 去除第一行全0项

    # 计算：recall、map、ndcg当前epoch的值
    arr = np.array([0.0 for _ in ranges])
    recalls, maps, ndcgs = arr.copy(), arr.copy(), arr.copy()
    hits, denominator_recalls = arr.copy(), np.sum(tes_masks)  # recall的分母，要预测这么些items
    for k in ranges:                            # 每次考察某个at值下的命中情况
        recoms = all_ranks[:, :at_nums[k]]      # 向每名user推荐这些
        # 逐行，得到recom_lst在test_lst里的命中情况，返回与recom_lst等长的0/1序列，1表示预测的该item在user_test里
        all_zero_ones = np.apply_along_axis(
            func1d=fun_hit_zero_one,
            axis=1,
            arr=np.array(zip(tes_buys_masks, recoms, tes_masks, append)))   # shape=(n_user, at_nums[k])
        hits[k] = np.sum(all_zero_ones)
        recalls[k] = 1.0 * np.sum(all_zero_ones) / denominator_recalls
        all_maps = np.apply_along_axis(
            func1d=fun_evaluate_map,
            axis=1,
            arr=np.array(zip(tes_buys_masks, all_zero_ones, tes_masks, append)))
        maps[k] = np.mean(all_maps)
        all_ndcgs = np.apply_along_axis(
            func1d=fun_evaluate_ndcg,
            axis=1,
            arr=np.array(zip(tes_buys_masks, all_zero_ones, tes_masks, append)))
        ndcgs[k] = np.mean(all_ndcgs)

    # 计算：hit_intervals的当前epoch值。取recall@30
    recoms = all_ranks[:, :p['intervals'][2]]
    recall_item_idxs = np.apply_along_axis(    # 找出recoms里出现在test里的那些items的标号
        func1d=fun_hit_recall_item_idx,
        axis=1,
        arr=np.array(zip(tes_buys_masks, recoms, tes_masks, append)))
    hit_intervals = fun_item_idx_to_intervals(
        recall_item_idxs,
        test_i_cou,
        p['intervals'])

    # 保存：recall/map/ndcg/hit_interval的最佳值
    # recall and intervals
    hit_cold_active = np.array([sum(hit_intervals[:2]), sum(hit_intervals[2:])])
    hit_intervals_cumsum = hit_intervals.cumsum()
    for k in ranges:
        if recalls[k] > best.best_recalls[k]:
            best.best_recalls[k] = recalls[k]
            best.best_epoch_recalls[k] = epoch
            if p['intervals'][2] == at_nums[k]:         # recall@30的区间划分
                best.best_recalls_cold_active = 1.0 * hit_cold_active / test_i_cold_active
                best.best_recalls_intervals_cumsum = 1.0 * hit_intervals_cumsum / test_i_intervals_cumsum
    # map and ndcg
    for k in ranges:
        if maps[k] > best.best_maps[k]:
            best.best_maps[k] = maps[k]
            best.best_epoch_maps[k] = epoch
        if ndcgs[k] > best.best_ndcgs[k]:
            best.best_ndcgs[k] = ndcgs[k]
            best.best_epoch_ndcgs[k] = epoch
    del all_upqs, all_ranks


def fun_predict_pop_random(
        p, best, all_upqs, all_ranks,
        tes_buys_masks, tes_masks,
        test_i_cou, test_i_intervals_cumsum, test_i_cold_active):

    append = [[0] for _ in np.arange(len(tes_buys_masks))]

    # 计算：AUC。
    if all_upqs is not None:
        auc = 1.0 * np.sum(all_upqs) / np.sum(tes_masks)    # 全部items

        auc_item_idxs = np.apply_along_axis(    # 找出all_upqs里标为true的那些items的标号
            func1d=fun_hit_auc_item_idx,
            axis=1,
            arr=np.array(zip(tes_buys_masks, all_upqs, tes_masks, append)))
        auc_intervals = fun_item_idx_to_intervals(
            auc_item_idxs,
            test_i_cou,
            p['intervals'])
        # 保存auc
        auc_cold_active = np.array([sum(auc_intervals[:2]), sum(auc_intervals[2:])])
        auc_intervals_cumsum = auc_intervals.cumsum()
        best.best_auc = auc
        best.best_auc_cold_active = 1.0 * auc_cold_active / test_i_cold_active
        best.best_auc_intervals_cumsum = 1.0 * auc_intervals_cumsum / test_i_intervals_cumsum

    at_nums = p['at_nums']
    ranges = range(len(at_nums))
    # 计算：recall、map、ndcg当前epoch的值
    arr = np.array([0.0 for _ in ranges])
    recalls, maps, ndcgs = arr.copy(), arr.copy(), arr.copy()
    hits, denominator_recalls = arr.copy(), np.sum(tes_masks)  # recall的分母，要预测这么些items
    for k in ranges:                            # 每次考察某个at值下的命中情况
        recoms = all_ranks[:, :at_nums[k]]      # 向每名user推荐这些
        # 得到predict的每行在test_buys每行里的命中情况，返回与predict等长的0/1序列，1表示预测的该item在user_test里
        all_zero_ones = np.apply_along_axis(
            func1d=fun_hit_zero_one,
            axis=1,
            arr=np.array(zip(tes_buys_masks, recoms, tes_masks, append)))   # shape=(n_user, at_nums[k])
        hits[k] = np.sum(all_zero_ones)
        recalls[k] = 1.0 * np.sum(all_zero_ones) / denominator_recalls
        all_maps = np.apply_along_axis(
            func1d=fun_evaluate_map,
            axis=1,
            arr=np.array(zip(tes_buys_masks, all_zero_ones, tes_masks, append)))
        maps[k] = np.mean(all_maps)
        all_ndcgs = np.apply_along_axis(
            func1d=fun_evaluate_ndcg,
            axis=1,
            arr=np.array(zip(tes_buys_masks, all_zero_ones, tes_masks, append)))
        ndcgs[k] = np.mean(all_ndcgs)

    # 计算：hit_intervals的当前epoch值。
    recoms = all_ranks[:, :p['intervals'][2]]
    hit_item_idxs = np.apply_along_axis(    # 找出recoms里出现在test里的那些items的标号
        func1d=fun_hit_recall_item_idx,
        axis=1,
        arr=np.array(zip(tes_buys_masks, recoms, tes_masks, append)))
    hit_intervals = fun_item_idx_to_intervals(
        hit_item_idxs,
        test_i_cou,
        p['intervals'])

    # 保存：保存auc/recall/map/ndcg/hit_interval的最佳值
    # recall
    hit_cold_active = np.array([sum(hit_intervals[:2]), sum(hit_intervals[2:])])
    hit_intervals_cumsum = hit_intervals.cumsum()
    for k in ranges:
        best.best_recalls[k] = recalls[k]
        if p['intervals'][2] == at_nums[k]:         # recall@30的区间划分
            best.best_recalls_cold_active = 1.0 * hit_cold_active / test_i_cold_active
            best.best_recalls_intervals_cumsum = 1.0 * hit_intervals_cumsum / test_i_intervals_cumsum
    # map and ndcg
    for k in ranges:
        best.best_maps[k] = maps[k]
        best.best_ndcgs[k] = ndcgs[k]


def fun_save_all_losses(
        path, model_name, epoch, losses,
        params, winh=None, winx=None):
    # 建立目录、文件名
    if os.path.exists(path):
        print('\t\tdir exists: {v1}'.format(v1=path))
    else:
        os.makedirs(path)
        print('\t\tdir is made: {v1}'.format(v1=path))
    now = datetime.datetime.now()
    time_str = now.strftime("%Y.%m.%d %H:%M:%S")    # 2017.08.18_11.22.27
    fil = os.path.join(
        path,
        model_name + '.txt')
    # 建立要保存的内容
    strs = \
        '\n' + model_name + \
        '\n\t' + 'epoch = ' + str(epoch) + \
        '\n\t' + 'winh, winx = ' + ', '.join([str(i) for i in [winh, winx]]) + \
        '\n\t' + 'alpha, lambda, lambda_ev, lambda_ae, zero = ' + ', '.join([str(i) for i in params]) + \
        '\n\t' + time_str + \
        '\n\t[' + ', '.join(losses) + ']\n'
    # 保存
    f = open(fil, 'a')
    f.write(strs)
    f.close()


@exe_time  # 放到待调用函数的定义的上一行
def main():
    print('... construct the evaluation program')


if '__main__' == __name__:
    main()
