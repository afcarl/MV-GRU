#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:

import datetime
import numpy as np
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
from theano.tensor.extra_ops import Unique
from GRU import GruBasic


def exe_time(func):
    def new_func(*args, **args2):
        name = func.__name__
        start = datetime.datetime.now()
        print("-- {%s} start: @ %ss" % (name, start.strftime("%Y.%m.%d_%H.%M.%S")))
        back = func(*args, **args2)
        end = datetime.datetime.now()
        print("-- {%s} start: @ %ss" % (name, start.strftime("%Y.%m.%d_%H.%M.%S")))
        print("-- {%s} end:   @ %ss" % (name, end.strftime("%Y.%m.%d_%H.%M.%S")))
        total = (end - start).total_seconds()
        print("-- {%s} total: @ %.2fs = %.2fh" % (name, total, total / 3600.0))
        return back
    return new_func


# 只是train相关的要做成mini-batch形式，其它的都和 Gru/MvGru 是一样的。
# 要做梯度归一化,即算出来的梯度除以batch size. 不除T.sum(tra_mask)，除以batch_size.
# ======================================================================================================================
class OwmGru(GruBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(OwmGru, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        # output
        rang = 0.5
        vy = uniform(-rang, rang, (n_item + 1, n_hidden))       # 和lt一样。
        self.vy = theano.shared(borrow=True, value=vy.astype(theano.config.floatX))
        self.params = [self.ui, self.wh, self.bi]       # self.lt单独进行更新。
        self.l2_sqr = (
            T.sum([T.sum(param ** 2) for param in [self.lt, self.vy]]) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_hidden):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda']
        ui, wh = self.ui, self.wh

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 20), n_hidden放在最后
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        # 输入端：只输入购买的商品即可。
        pidxs, qidxs = T.imatrix(), T.imatrix()         # TensorType(int32, matrix)
        xps = self.lt[pidxs]       # shape((actual_batch_size, seq_length, n_in))
        xps = xps.dimshuffle(1, 0, 2)     # shape=(seq_length, batch_size, n_in)

        uiq_ps = Unique(False, False, False)(pidxs)  # 再去重
        uiq_x = self.lt[uiq_ps]

        # 输出端：h*w 得到score
        yps, yqs = self.vy[pidxs], self.vy[qidxs]
        yps, yqs = yps.dimshuffle(1, 0, 2), yqs.dimshuffle(1, 0, 2)

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_y = self.vy[uiq_pqs]                    # 相应的items特征

        """
        输入t时刻正负样本、t-1时刻隐层，计算当前隐层、当前损失. 公式里省略了时刻t
        # 根据性质：T.dot((m, n), (n, ))得到shape=(m, )，且是矩阵每行与(n, )相乘
            # GRU
            z = sigmoid(ux_z * xp + wh_z * h_pre1)
            r = sigmoid(ux_r * xp + wh_r * h_pre1)
            c = tanh(ux_c * xp + wh_c * (r 点乘 h_pre1))
            h = z * h_pre1 + (1.0 - z) * c
        # 根据性质：T.dot((n, ), (n, ))得到scalar
            upq  = h_pre1 * (xp - xq)
            loss = log(1.0 + e^(-upq))
        """
        def recurrence(xp_t, yp_t, yq_t, mask_t, h_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            z_r = sigmoid(T.dot(ui[:2], xp_t.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])   # shape=(2, 20, n)
            z, r = z_r[0].T, z_r[1].T                           # shape=(n, 20)
            c = tanh(T.dot(ui[2], xp_t.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])    # shape=(20, n)
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T     # shape=(n, 20)
            # 偏好误差
            upq_t = T.sum(h_t_pre1 * (yp_t - yq_t), axis=1)     # shape=(n, )
            loss_t = T.log(sigmoid(upq_t))                      # shape=(n, )
            loss_t *= mask_t                            # 只在损失这里乘一下0/1向量就可以了
            return [h_t, loss_t]                        # shape=(n, 20), (n, )
        [h, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[xps, yps, yqs, mask],
            outputs_info=[h0, None],
            n_steps=seq_length)     # 保证只循环到最长有效位

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [xps, ui, wh, yps, yqs]]) +
            T.sum([T.sum(par ** 2) for par in [bi]]) / actual_batch_size)
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_ps])
        update_y = T.set_subtensor(uiq_y, uiq_y - lr * T.grad(seq_costs, self.vy)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        seq_updates.append((self.vy, update_y))
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        # givens给数据
        start_end = T.ivector()     # int32
        self.seq_train = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],   # T.ivector()类型是 TensorType(int32, vector)
                tra_mask: self.tra_masks[start_end]})

    def train(self, idxs):
        return self.seq_train(idxs)

    def update_trained_items(self):
        # 更新最终的items表达
        # 注意：这里的trained_items，只用于推荐前的重新计算用户表达用。
        lt = self.lt.get_value(borrow=True)    # self.vy是shared，用get_value()。# shape=(n_item+1, 60)
        self.trained_items.set_value(np.asarray(lt, dtype=theano.config.floatX), borrow=True)     # update

    def update_trained_items_by_vy(self):
        # 更新最终的items表达
        # 注意：这里的trained_items，是真正最后用于做推荐的items表达。
        vy = self.vy.get_value(borrow=True)    # self.vy是shared，用get_value()。# shape=(n_item+1, 60)
        self.trained_items.set_value(np.asarray(vy, dtype=theano.config.floatX), borrow=True)     # update


# ======================================================================================================================
class OwmMvGruCon(GruBasic):
    """
    1. 图文特征降维
    2. 和latent特征做拼接. Concatenate
    3. 送入1个GRU
    """
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden,
                 n_img, n_txt, fea_img, fea_txt):
        super(OwmMvGruCon, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.fi = theano.shared(borrow=True, value=np.asarray(fea_img, dtype=theano.config.floatX))  # shape=(n, 1024)
        self.ft = theano.shared(borrow=True, value=np.asarray(fea_txt, dtype=theano.config.floatX))  # shape=(n, 100)
        # 其它参数
        rang = 0.5
        mi = uniform(-rang, rang, (n_item + 1, n_in))   # 图像的低维特征，和 lt 一一对应。
        mt = uniform(-rang, rang, (n_item + 1, n_in))   # 文本的低维特征，
        ei = uniform(-rang, rang, (n_in, n_img))                # shape=(20, 1024)
        vt = uniform(-rang, rang, (n_in, n_txt))                # shape=(20, 100)
        self.mi = theano.shared(borrow=True, value=mi.astype(theano.config.floatX))
        self.mt = theano.shared(borrow=True, value=mt.astype(theano.config.floatX))
        self.ei = theano.shared(borrow=True, value=ei.astype(theano.config.floatX))
        self.vt = theano.shared(borrow=True, value=vt.astype(theano.config.floatX))
        # output
        vy = uniform(-rang, rang, (n_item + 1, n_hidden))       # 和lt一样。
        self.vy = theano.shared(borrow=True, value=vy.astype(theano.config.floatX))
        self.params = [
            self.ui, self.wh, self.bi,                  # self.lt/vy单独进行更新。
            self.ei, self.vt]
        self.l2_sqr = (
            T.sum([T.sum(param ** 2) for param in [self.lt, self.vy]]) +
            T.sum([T.sum(param ** 2) for param in self.params[:4]]))
        self.l2_ev = (
            T.sum(self.ei ** 2) +
            T.sum(self.vt ** 2))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr +
            0.5 * self.alpha_lambda[2] * self.l2_ev)
        self.__theano_train__(n_hidden, n_img, n_txt)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_hidden, n_img, n_txt):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda', 'lambda_ev']
        ui, wh = self.ui, self.wh
        ei, vt = self.ei, self.vt

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 20), n_hidden放在最后
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        pidxs, qidxs = T.imatrix(), T.imatrix()         # TensorType(int32, matrix)
        xps = self.lt[pidxs]       # shape((actual_batch_size, seq_length, n_in))
        ips = self.fi[pidxs]       # shape((actual_batch_size, seq_length, n_img))
        tps = self.ft[pidxs]       # shape((actual_batch_size, seq_length, n_txt))
        xps = xps.dimshuffle(1, 0, 2)     # shape=(seq_len, batch_size, n_in)
        ips = ips.dimshuffle(1, 0, 2)
        tps = tps.dimshuffle(1, 0, 2)

        uiq_ps = Unique(False, False, False)(pidxs)  # 再去重
        uiq_x = self.lt[uiq_ps]

        # 输出端：h*w 得到score
        yps, yqs = self.vy[pidxs], self.vy[qidxs]
        yps, yqs = yps.dimshuffle(1, 0, 2), yqs.dimshuffle(1, 0, 2)

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_y = self.vy[uiq_pqs]                    # 相应的items特征

        """
        输入t时刻正负样本、t-1时刻隐层，计算当前隐层、当前损失. 公式里省略了时刻t
        # 根据性质：T.dot((m, n), (n, ))得到shape=(m, )，且是矩阵每行与(n, )相乘
            # GRU
            z = sigmoid(ux_z * xp + wh_z * h_pre1)
            r = sigmoid(ux_r * xp + wh_r * h_pre1)
            c = tanh(ux_c * xp + wh_c * (r 点乘 h_pre1))
            h = z * h_pre1 + (1.0 - z) * c
        # 根据性质：T.dot((n, ), (n, ))得到scalar
            upq  = h_pre1 * (xp - xq)
            loss = log(1.0 + e^(-upq))
        """
        def recurrence(xp_t, ip_t, tp_t, yp_t, yq_t, mask_t, h_t_pre1):
            # item表达
            mip_t, mtp_t = T.dot(ip_t, ei.T), T.dot(tp_t, vt.T)     # shape=(n, 20)
            p_t = T.concatenate((xp_t, mip_t, mtp_t), axis=1)       # shape=(n, 60)
            # 隐层计算
            z_r = sigmoid(T.dot(ui[:2], p_t.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])
            z, r = z_r[0].T, z_r[1].T           # shape=(n, 40)
            c = tanh(T.dot(ui[2], p_t.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T
            # 偏好误差
            upq_t = T.sum(h_t_pre1 * (yp_t - yq_t), axis=1)   # shape=(n, )
            loss_t = T.log(sigmoid(upq_t))                  # shape=(n, )
            loss_t *= mask_t
            return [h_t, loss_t]  # shape=(n, 20), (n, ), (n, )
        [h, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[xps, ips, tps, yps, yqs, mask],
            outputs_info=[h0, None],
            n_steps=seq_length,
            truncate_gradient=-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        l2_ev = self.alpha_lambda[2]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [xps, ui, wh, yps, yqs]]) +
            T.sum([T.sum(par ** 2) for par in [bi]]) / actual_batch_size)
        seq_l2_ev = (
            T.sum([T.sum(par ** 2) for par in [ei, vt]]))
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq +
            0.5 * l2_ev * seq_l2_ev)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_ps])
        update_y = T.set_subtensor(uiq_y, uiq_y - lr * T.grad(seq_costs, self.vy)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        seq_updates.append((self.vy, update_y))
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        # givens给数据
        start_end = T.ivector()
        self.seq_train = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],   # T.ivector()类型是 TensorType(int32, vector)
                tra_mask: self.tra_masks[start_end]})

    def train(self, idxs):
        return self.seq_train(idxs)

    def update_trained_items(self):
        # 获取图文融合表达。先用.eval()获得十进制数字，再用set_value()对shared变量做更新。
        mi, mt = T.dot(self.fi, self.ei.T), T.dot(self.ft, self.vt.T)   # shape=(n_item+1, 20)
        mi, mt = mi.eval(), mt.eval()
        self.mi.set_value(np.asarray(mi, dtype=theano.config.floatX), borrow=True)
        self.mt.set_value(np.asarray(mt, dtype=theano.config.floatX), borrow=True)
        # 更新最终的items表达。
        # 注意：这里的trained_items，只用于推荐前的重新计算用户表达用。
        items = T.concatenate((self.lt, self.mi, self.mt), axis=1)      # shape=(n_item+1, 60)
        items = items.eval()
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)

    def update_trained_items_by_vy(self):
        # 更新最终的items表达
        # 注意：这里的trained_items，是真正最后用于做推荐的items表达。
        vy = self.vy.get_value(borrow=True)    # self.vy是shared，用get_value()。# shape=(n_item+1, 60)
        self.trained_items.set_value(np.asarray(vy, dtype=theano.config.floatX), borrow=True)     # update


@exe_time  # 放到待调用函数的定义的上一行
def main():
    pass


if '__main__' == __name__:
    main()
