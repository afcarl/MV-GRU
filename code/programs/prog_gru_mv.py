#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:   纯RNN、GRU程序


from __future__ import print_function
from collections import OrderedDict     # 按输入的顺序构建字典
import time
import numpy as np
import os
import random
import datetime
from public.GRU import Gru, Lstm
from public.GRU_MV import MvGru1Unit, MvGru2Units, MvGruCon, MvGruFusion
from public.Global_Best import GlobalBest
from public.Valuate import fun_predict_auc_recall_map_ndcg, fun_save_all_losses
from public.Load_Data import load_data, fun_data_buys_masks, load_img_txt
from public.Load_Data import fun_random_neg_tra, fun_random_neg_tes
__docformat__ = 'restructedtext en'

# WHOLE = 'E:/Projects/datasets/'
WHOLE = '/home/cuiqiang/2_Datasets/'
# WHOLE = '/home/wushu/usr_cuiqiang/2_Datasets/'
# WHOLE = '/mnt/dev_sdb1/cuiqiang/2_Datasets/'
PATH_a5 = os.path.join(WHOLE, 'amazon_users_5_100/')
PATH_t5 = os.path.join(WHOLE, 'taobao_users_5_100/')
PATH_t55 = os.path.join(WHOLE, 'taobao_users_5_100_5000hang/')
PATH = PATH_a5     # 109上先试试t55，确保正常运行。


def exe_time(func):
    def new_func(*args, **args2):
        name = func.__name__
        start = datetime.datetime.now()
        print("-- {%s} start: @ %ss" % (name, start.strftime("%Y.%m.%d %H:%M:%S")))
        back = func(*args, **args2)
        end = datetime.datetime.now()
        print("-- {%s} start: @ %ss" % (name, start.strftime("%Y.%m.%d %H:%M:%S")))
        print("-- {%s} end:   @ %ss" % (name, end.strftime("%Y.%m.%d %H:%M:%S")))
        total = (end - start).total_seconds()
        print("-- {%s} total: @ %.2fs = %.2fh" % (name, total, total / 3600.0))
        return back
    return new_func


# ================================================================================================
# missing/denoising模式联合运行。


class Params(object):
    def __init__(self, p=None):
        """
        构建模型参数，加载数据
            把前80%分为6:2用作train和valid，来选择超参数, 不用去管剩下的20%.
            把前80%作为train，剩下的是test，把valid时学到的参数拿过来跑程序.
            valid和test部分，程序是一样的，区别在于送入的数据而已。
        :param p: 一个标示符，没啥用
        :return:
        """
        global PATH
        # 1. 建立各参数。要调整的地方都在 p 这了，其它函数都给写死。
        if not p:
            v = 0                       # 写1就是valid, 写0就是test
            assert 0 == v or 1 == v     # no other case
            p = OrderedDict(
                [
                    ('dataset',             'user_buys.txt'),
                    ('fea_image',           'normalized_features_image/'),
                    ('fea_text',            'normalized_features_text/'),
                    ('mode',                'valid' if 1 == v else 'test'),
                    ('split',               0.8),               # valid: 6/2/2。test: 8/2.
                    ('at_nums',             [10, 20, 30, 50]),  # 5， 15
                    ('intervals',           [2, 10, 30]),       # 以次数2为间隔，分为10个区间. 计算auc/recall@30上的. 换为10
                    ('epochs',              30 if 'taobao' in PATH else 50),
                    ('alpha',               0.1),

                    ('latent_size',         [20, 1024, 100]),
                    ('lambda',              0.001),         # 要不要self.lt和self.ux/wh/bi用不同的lambda？
                    ('lambda_ev',           0.0),           # 图文降维局矩阵的。就是这个0.0
                    ('lambda_ae',           0.001),         # 重构误差的。

                    ('train_fea_zero',      0.0),   # 0.1、0.2/0.3
                    ('mvgru',               0),     # 0:gru,
                                                    # 1:mv-gru-1unit, 2:mv-gru-2units, 3:mv-gru-con, 4:mv-gru-fusion

                    ('batch_size_train',    4),     # size大了之后性能下降非常严重
                    ('batch_size_test',     768),   # user*item矩阵太大，要多次计算。a5下亲测768最快。
                ])
            for i in p.items():
                print(i)
            assert p['mode'] in ['valid', 'test']
            # assert (p['missing'] + p['denoising']) in [0, 1]

        # 2. 加载数据
        # 因为train/set里每项的长度不等，无法转换为完全的(n, m)矩阵样式，所以shared会报错.
        [(user_num, item_num), aliases_dict,
         (test_i_cou, test_i_intervals_cumsum, test_i_cold_active),
         (tra_buys, tes_buys), (set_tra, set_tes)] = \
            load_data(os.path.join(PATH, p['dataset']),
                      p['mode'], p['split'], p['intervals'])
        # 正样本加masks
        tra_buys_masks, tra_masks = fun_data_buys_masks(tra_buys, tail=[item_num])          # 预测时算用户表达用
        tes_buys_masks, tes_masks = fun_data_buys_masks(tes_buys, tail=[item_num])          # 预测时用
        # 负样本加masks
        tra_buys_neg_masks = fun_random_neg_tra(item_num, tra_buys_masks)   # 训练时用（逐条、mini-batch均可）
        tes_buys_neg_masks = fun_random_neg_tes(item_num, tra_buys_masks, tes_buys_masks)   # 预测时用

        # 3. 创建类变量
        self.p = p
        self.user_num, self.item_num = user_num, item_num
        self.aliases_dict = aliases_dict
        self.tic, self.tiic, self.tica = test_i_cou, test_i_intervals_cumsum, test_i_cold_active
        self.tra_buys, self.tes_buys = tra_buys, tes_buys
        self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks = tra_buys_masks, tra_masks, tra_buys_neg_masks
        self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks = tes_buys_masks, tes_masks, tes_buys_neg_masks
        self.set_tra, self.set_tes = set_tra, set_tes

    def build_model_mini_batch(self, flag):
        """
        建立模型对象
        :param flag: 参数变量、数据
        :return:
        """
        print('building the model ...')
        assert flag in [0, 1, 2, 3, 4]
        p = self.p
        size = p['latent_size'][0]
        if 0 == flag:        # Gru
            size_total = 1
            model = Lstm(
                train=[self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks],
                test =[self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks],
                alpha_lambda=[p['alpha'], p['lambda']],
                n_user=self.user_num,
                n_item=self.item_num,
                n_in=size,
                n_hidden=size * 1)
        else:
            # 加载图文数据
            fea_img = load_img_txt('Image', os.path.join(PATH, p['fea_image']),
                                   self.item_num, p['latent_size'][1], self.aliases_dict)
            fea_txt = load_img_txt('Text', os.path.join(PATH, p['fea_text']),
                                   self.item_num, p['latent_size'][2], self.aliases_dict)
            # 把test里取1/3items的图特征置为0，1/3items的文特征置为0，1/3items的特征不动。
            set_tes, len_tes = self.set_tes, len(self.set_tes)
            no_img = set(random.sample(set_tes, len_tes // 3))
            set_tes -= no_img
            no_txt = set(random.sample(set_tes, len_tes // 3))
            fea_img_ct = fea_img.copy()     # 不copy()一下的话，改变fea_img_ct后，fea_img也会改变。
            fea_txt_ct = fea_txt.copy()
            print(list(no_img)[:3], np.asarray(list(no_txt))[:3])
            fea_img_ct[np.asarray(list(no_img))] = 0.0
            fea_txt_ct[np.asarray(list(no_txt))] = 0.0
            if 1 == flag:
                size_total = 2
                model = MvGru1Unit(
                    train=[self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks],
                    test= [self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks],
                    alpha_lambda=[p['alpha'], p['lambda'], p['lambda_ev'], p['lambda_ae'], p['train_fea_zero']],
                    n_user=self.user_num,
                    n_item=self.item_num,
                    n_in=size,
                    n_hidden=size * 2,      # 1个GRU单元，n_hidden = 2*n_in。2特征拼接。
                    n_img=p['latent_size'][1],
                    n_txt=p['latent_size'][2],
                    fea_img=fea_img,
                    fea_txt=fea_txt,
                    fea_img_ct=fea_img_ct,
                    fea_txt_ct=fea_txt_ct)
            elif 2 == flag:
                size_total = 2
                model = MvGru2Units(
                    train=[self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks],
                    test= [self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks],
                    alpha_lambda=[p['alpha'], p['lambda'], p['lambda_ev'], p['lambda_ae'], p['train_fea_zero']],
                    n_user=self.user_num,
                    n_item=self.item_num,
                    n_in=size,
                    n_hidden=size * 1,      # 2个GRU单元，n_hidden = n_in。2特征分离。
                    n_img=p['latent_size'][1],
                    n_txt=p['latent_size'][2],
                    fea_img=fea_img,
                    fea_txt=fea_txt,
                    fea_img_ct=fea_img_ct,
                    fea_txt_ct=fea_txt_ct)
            elif 3 == flag:
                size_total = 3
                model = MvGruCon(
                    train=[self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks],
                    test= [self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks],
                    alpha_lambda=[p['alpha'], p['lambda'], p['lambda_ev']],
                    n_user=self.user_num,
                    n_item=self.item_num,
                    n_in=size,
                    n_hidden=size * 3,      # 1个GRU单元，n_hidden = 3*n_in。3特征拼接。
                    n_img=p['latent_size'][1],
                    n_txt=p['latent_size'][2],
                    fea_img=fea_img,
                    fea_txt=fea_txt,
                    fea_img_ct=fea_img_ct,
                    fea_txt_ct=fea_txt_ct)
            else:
                size_total = 2
                model = MvGruFusion(
                    train=[self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks],
                    test= [self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks],
                    alpha_lambda=[p['alpha'], p['lambda'], p['lambda_ev']],
                    n_user=self.user_num,
                    n_item=self.item_num,
                    n_in=size,
                    n_hidden=size * 2,      # 1个GRU单元，n_hidden = 2*n_in。2特征拼接。
                    n_img=p['latent_size'][1],
                    n_txt=p['latent_size'][2],
                    fea_img=fea_img,
                    fea_txt=fea_txt,
                    fea_img_ct=fea_img_ct,
                    fea_txt_ct=fea_txt_ct)
        model_name = model.__class__.__name__
        print('\t the current Class name is: {val}'.format(val=model_name))
        if (model_name in ['Gru', 'MvGruCon', 'MvGruFusion']) and (p['train_fea_zero'] != 0):
            print('Error: if model is Gru or MvGruCon or MvGruFusion, p[\'train_fea_zero\'] should be 0.0.')
            raise
        return model, model_name, size_total * size

    def compute_start_end(self, flag):
        """
        获取mini-batch的各个start_end(np.array类型，一组连续的数值)
        :param flag: 'train', 'test'
        :return: 各个start_end组成的list
        """
        assert 'train' == flag or 'test' == flag or 'auc' == flag
        if 'train' == flag:
            size = self.p['batch_size_train']
        elif 'test' == flag:
            size = self.p['batch_size_test']        # test: top-k
        else:
            size = self.p['batch_size_test'] * 10   # test: auc
        user_num = self.user_num
        rest = (user_num % size) > 0   # 能整除：rest=0。不能整除：rest=1，则多出来一个小的batch
        n_batches = np.minimum(user_num // size + rest, user_num)
        batch_idxs = np.arange(n_batches, dtype=np.int32)
        starts_ends = []
        for bidx in batch_idxs:
            start = bidx * size
            end = np.minimum(start + size, user_num)   # 限制标号索引不能超过user_num
            start_end = np.arange(start, end, dtype=np.int32)
            starts_ends.append(start_end)
        return batch_idxs, starts_ends


def train_valid_or_test():
    """
    主程序
    :return:
    """
    global PATH
    # 建立参数、数据、模型、模型最佳值
    pas = Params()
    p = pas.p
    model, model_name, size_total = pas.build_model_mini_batch(flag=p['mvgru'])
    best_denoise = GlobalBest(at_nums=p['at_nums'], intervals=p['intervals'])   # 存放最优数据
    best_missing = GlobalBest(at_nums=p['at_nums'], intervals=p['intervals'])
    batch_idxs_tra, starts_ends_tra = pas.compute_start_end(flag='train')
    _, starts_ends_tes = pas.compute_start_end(flag='test')
    _, starts_ends_auc = pas.compute_start_end(flag='auc')

    # 直接取出来部分变量，后边就不用加'pas.'了。
    user_num, item_num = pas.user_num, pas.item_num
    tra_buys_masks, tes_buys_masks = pas.tra_buys_masks, pas.tes_buys_masks
    tes_masks = pas.tes_masks
    test_i_cou, test_i_intervals_cumsum, test_i_cold_active = pas.tic, pas.tiic, pas.tica
    del pas

    # 主循环
    losses = []
    times0, times1, times2, times3 = [], [], [], []
    for epoch in np.arange(p['epochs']):
        print("Epoch {val} ==================================".format(val=epoch))
        # 每次epoch，都要重新选择负样本。都要把数据打乱重排，这样会以随机方式选择样本计算梯度，可得到精确结果
        if epoch > 0:       # epoch=0的负样本已在循环前生成，且已用于类的初始化
            tra_buys_neg_masks = fun_random_neg_tra(item_num, tra_buys_masks)
            tes_buys_neg_masks = fun_random_neg_tes(item_num, tra_buys_masks, tes_buys_masks)
            model.update_neg_masks(tra_buys_neg_masks, tes_buys_neg_masks)

        # --------------------------------------------------------------------------------------------------------------
        print("\tTraining ...")
        t0 = time.time()
        loss = 0.
        random.seed(str(123 + epoch))
        random.shuffle(batch_idxs_tra)      # 每个epoch都打乱batch_idx输入顺序
        for bidx in batch_idxs_tra:
            start_end = starts_ends_tra[bidx]
            random.shuffle(start_end)       # 打乱batch内的indexes
            loss += model.train(start_end)
        rnn_l2_sqr = model.l2.eval()            # model.l2是'TensorVariable'，无法直接显示其值
        print('\t\tsum_loss = {val} = {v1} + {v2}'.format(val=loss + rnn_l2_sqr, v1=loss, v2=rnn_l2_sqr))
        losses.append('%0.2f' % (loss + rnn_l2_sqr))
        t1 = time.time()
        times0.append(t1 - t0)

        # --------------------------------------------------------------------------------------------------------------
        print("\tPredicting ...")
        # 计算：所有用户、商品的表达
        model.update_trained_items()    # 要先运行这个更新items特征。对于MV-GRU，这里会先算出来图文融合特征。
        all_hus = np.array([[0.0 for _ in np.arange(size_total)]])          # 初始shape=(1, 20/40)
        for start_end in starts_ends_tes:
            sub_all_hus = model.predict(start_end)
            all_hus = np.concatenate((all_hus, sub_all_hus))
        all_hus = np.delete(all_hus, 0, axis=0)         # 去除第一行全0项,  # shape=(n_user, n_hidden)
        model.update_trained_users(all_hus)
        t2 = time.time()
        times1.append(t2 - t1)

        # denoise模式：test用完整数据。
        fun_predict_auc_recall_map_ndcg(
            p, model, best_denoise, epoch, starts_ends_auc, starts_ends_tes,
            tes_buys_masks, tes_masks,
            test_i_cou, test_i_intervals_cumsum, test_i_cold_active)
        best_denoise.fun_print_best(epoch)   # 每次都只输出当前最优的结果
        t3 = time.time()
        times2.append(t3-t2)
        print('\tdenoise: avg. time (train, user, test): %0.0fs,' % np.average(times0),
              '%0.0fs,' % np.average(times1), '%0.0fs |' % np.average(times2),
              datetime.datetime.now().strftime("%Y.%m.%d %H:%M"),
              '| model: %s' % model_name,
              '| lam: %s' % ', '.join([str(lam) for lam in [p['lambda'], p['lambda_ev'], p['lambda_ae']]]),
              '| train_fea_zero: %0.1f' % p['train_fea_zero'])

        if 'MvGru' in model_name:
            # missing模式：test用缺失数据。
            model.update_trained_items2_corrupted_test_data()   # 注意：missing下的test data是有破损的。
            fun_predict_auc_recall_map_ndcg(
                p, model, best_missing, epoch, starts_ends_auc, starts_ends_tes,
                tes_buys_masks, tes_masks,
                test_i_cou, test_i_intervals_cumsum, test_i_cold_active)
            best_missing.fun_print_best(epoch)   # 每次都只输出当前最优的结果
            t4 = time.time()
            times3.append(t4-t3)
            print('\tmissing: avg. time (train, user, test): %0.0fs,' % np.average(times0),
                  '%0.0fs,' % np.average(times1), '%0.0fs |' % np.average(times3),
                  datetime.datetime.now().strftime("%Y.%m.%d %H:%M"),
                  '| model: %s' % model_name,
                  '| lam: %s' % ', '.join([str(lam) for lam in [p['lambda'], p['lambda_ev'], p['lambda_ae']]]),
                  '| train_fea_zero: %0.1f' % p['train_fea_zero'])

        # --------------------------------------------------------------------------------------------------------------
        # 保存epoch=29/49时的最优值。
        if epoch == p['epochs'] - 1:
            print("\t-----------------------------------------------------------------")
            print("\tBest saving ...")
            path = os.path.join(os.path.split(__file__)[0], '..', 'Results_best_values', PATH.split('/')[-2])
            best_denoise.fun_save_best(
                path, model_name + ' - denoise', epoch, [p['batch_size_train'], p['batch_size_test']],
                [p['alpha'], p['lambda'], p['lambda_ev'], p['lambda_ae'], p['train_fea_zero']])
            if 'MvGru' in model_name:
                best_missing.fun_save_best(
                    path, model_name + ' - missing', epoch, [p['batch_size_train'], p['batch_size_test']],
                    [p['alpha'], p['lambda'], p['lambda_ev'], p['lambda_ae'], p['train_fea_zero']])

        # --------------------------------------------------------------------------------------------------------------
        # 保存所有的损失值。
        if epoch == p['epochs'] - 1:
            print("\tLoss saving ...")
            path = os.path.join(os.path.split(__file__)[0], '..', 'Results_alpha_0.1_loss', PATH.split('/')[-2])
            fun_save_all_losses(
                path, model_name, epoch, losses,
                [p['alpha'], p['lambda'], p['lambda_ev'], p['lambda_ae'], p['train_fea_zero']])

    for i in p.items():
        print(i)
    print('\t the current Class name is: {val}'.format(val=model_name))


@exe_time  # 放到待调用函数的定义的上一行
def main():
    # pas = Params()
    train_valid_or_test()


if '__main__' == __name__:
    main()
