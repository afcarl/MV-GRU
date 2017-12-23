#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:

import time
import numpy as np
import os
import datetime


def exe_time(func):
    def new_func(*args, **args2):
        t0 = time.time()
        print("-- @%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("-- @%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("-- @%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return new_func


class GlobalBest(object):
    # 全局评价指标放在这里。保存最优的值、对应的epoch
    def __init__(self, at_nums, intervals):
        """
        :param at_nums:     [5, 10, 15, 20, 30, 50]
        :param intervals:   [2, 10, 30]
        :return:
        """
        self.intervals = intervals
        ranges = np.arange(len(at_nums))
        arf_val = np.array([0.0 for _ in ranges])
        ari_epo = np.array([0 for _ in ranges])
        inters = np.array([0.0 for _ in np.arange(intervals[1])])   # 10个区间
        cold_active = np.array([0.0, 0.0])

        self.best_auc = 0.0
        self.best_recalls = arf_val.copy()
        self.best_maps = arf_val.copy()
        self.best_ndcgs = arf_val.copy()

        self.best_epoch_auc = 0
        self.best_epoch_recalls = ari_epo.copy()
        self.best_epoch_maps = ari_epo.copy()
        self.best_epoch_ndcgs = ari_epo.copy()

        self.best_auc_intervals_cumsum = inters.copy()
        self.best_auc_cold_active = cold_active.copy()

        self.best_recalls_intervals_cumsum = inters.copy()
        self.best_recalls_cold_active = cold_active.copy()

    def fun_obtain_best(self, epoch):
        """
        :param epoch:
        :return: 由最优值组成的字符串
        """
        def truncate4(x):
            """
            把输入截断为六位小数
            :param x:
            :return: 返回一个字符串而不是list里单项是字符串，
            这样打印的时候就严格是小数位4维，并且没有字符串的起末标识''
            """
            return ', '.join(['%0.4f' % i for i in x])
        amp = 100
        one = '\t'
        two, three = one * 2, one * 3
        a = one + '-----------------------------------------------------------------'
        # 指标值、最佳的epoch
        b = one + 'All values is the "best * {v1}" on epoch {v2}:'.format(v1=amp, v2=epoch)
        c = two + 'AUC       = [{val}], '.format(val=truncate4([self.best_auc * amp])) + \
            one + '{val}'.format(val=[self.best_epoch_auc])
        d = two + 'Recall    = [{val}], '.format(val=truncate4(self.best_recalls * amp)) + \
            one + '{val}'.format(val=self.best_epoch_recalls)
        e = two + 'MAP       = [{val}], '.format(val=truncate4(self.best_maps * amp)) + \
            one + '{val}'.format(val=self.best_epoch_maps)
        f = two + 'NDCG      = [{val}], '.format(val=truncate4(self.best_ndcgs * amp)) + \
            one + '{val}'.format(val=self.best_epoch_ndcgs)
        # # 分区间考虑。调参时去掉这块。
        g = one + 'cold_active | Intervals_cumsum:'
        h = two + 'AUC       = [{v1}], [{v2}]'. \
            format(v1=truncate4(self.best_auc_cold_active * amp),
                   v2=truncate4(self.best_auc_intervals_cumsum * amp))
        i = two + 'Recall@{v1} = [{v2}], [{v3}]'. \
            format(v1=self.intervals[2],
                   v2=truncate4(self.best_recalls_cold_active * amp),
                   v3=truncate4(self.best_recalls_intervals_cumsum * amp))
        return '\n'.join([a, b, c, d, e, f, g, h, i])

    def fun_print_best(self, epoch):
        # 输出最优值
        print(self.fun_obtain_best(epoch))

    def fun_save_best(self, path, model_name, epoch, sizes, params):
        """
        保存最优值
        :param path:
        :param model_name:
        :param epoch:
        :param params:
        :return: 将第epoch个的最优值保存到文件里
        """
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
            '\n\t' + time_str + \
            '\n\t' + 'batch size train, test = ' + ', '.join([str(i) for i in sizes]) + \
            '\n\t' + 'alpha, lambda, lambda_ev, lambda_ae, zero = ' + ', '.join([str(i) for i in params]) + \
            '\n' + self.fun_obtain_best(epoch) + '\n'
        # 保存
        f = open(fil, 'a')
        f.write(strs)
        f.close()


@exe_time  # 放到待调用函数的定义的上一行
def main():
    obj = GlobalBest(
        at_nums=[5, 10, 20, 30, 50, 100],
        intervals=[2, 10, 30])
    print('创建类对象后，读取实例变量值')
    print(obj.best_auc)     # 建立类对象后，可读取实例变量

    obj.best_auc = 70.3
    print('直接复制操作修改实例变量，再读取查看效果')
    print(obj.best_auc)     # 实例变量可直接修改


if '__main__' == __name__:
    main()


