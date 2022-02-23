###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
from __future__ import print_function
import torch
import os
import random
from tqdm import tqdm
import numpy as np
import time
import os
import shutil
import json
from DeepKernelGPHelpers import totorch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr
import datetime
from MetaD2A_nas_bench_201.utils import load_graph_config, decode_igraph_to_NAS_BENCH_201_string
from MetaD2A_nas_bench_201.utils import Log, get_log
from MetaD2A_nas_bench_201.utils import load_model, save_model, mean_confidence_interval

from MetaD2A_nas_bench_201.loader import get_meta_train_loader, get_meta_test_loader, MetaTestDataset
from encoder_FSBO import EncoderFSBO as PredictorModel
from MetaD2A_nas_bench_201.nas_bench_201 import train_single_model
from scipy.stats import norm

def EI(incumbent, mu, stddev, support, return_score=False):
    with np.errstate(divide='warn'):
        imp = mu - incumbent
        Z = imp / stddev
        score = imp * norm.cdf(Z) + stddev * norm.pdf(Z)
    if not return_score:
        score[support] = 0
        return np.argmax(score)
    else:
        # score[support] = 0
        return score

class Predictor:
    def __init__(self, args, lr=1e-1, dgp_arch=[32, 32, 32, 32], bohb=False):
        self.args = args
        self.batch_size = args.batch_size
        self.data_path = args.data_path
        self.num_sample = args.num_sample
        self.max_epoch = args.max_epoch
        self.save_epoch = args.save_epoch
        self.model_path = args.model_path
        self.save_path = args.save_path
        self.model_name = args.model_name
        self.test = args.test
        self.device = torch.device("cpu") # "cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_corr_dict = {'corr': -1, 'epoch': -1}
        self.train_arch = args.train_arch

        graph_config = load_graph_config(
            args.graph_data_name, args.nvt, args.data_path)

        self.model = PredictorModel(args, graph_config, dgp_arch=dgp_arch)
        # self.model.to(self.device)

        if self.test or bohb:
            self.data_name = args.data_name
            self.num_class = args.num_class
            self.load_epoch = args.load_epoch
            # if self.test:
                # load_model(self.model, self.model_path, load_max_pt='ckpt_max_corr.pt')
                # self.model.randomly_init_deepgp()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min',
                      factor=0.1, patience=1000, verbose=True)
        # torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer , T_max=args.max_epoch,
        #                                                            eta_min=1e-5)
        self.mtrloader = get_meta_train_loader(
            self.batch_size, self.data_path, self.num_sample, is_pred=True)

        self.acc_mean = self.mtrloader.dataset.mean
        self.acc_std = self.mtrloader.dataset.std

        self.mtrlog = Log(self.args, open(os.path.join(
            self.save_path, self.model_name, 'meta_train_predictor.log'), 'w'))
        self.mtrlog.print_args()

    def forward(self, x, arch, labels=None, train=False, matrix=False):
        D_mu = self.model.set_encode(x.to(self.device))
        G_mu = self.model.graph_encode(arch, matrix=matrix)
        y_pred, y_dist = self.model.predict(D_mu, G_mu, labels=labels, train=train)
        return y_pred, y_dist

    def forward_finetune(self, z, labels=None, train=True, matrix=False):
        y_pred, y_dist = self.model.predict_finetune(z, labels=labels, train=True)
        return y_pred, y_dist

    def get_mu_and_std(self, x_support, y_support, x_query, y_query):
        self.model.eval()
        mu, std = self.model.get_mu_and_std(x_support, y_support, x_query, y_query)
        return mu, std

    def finetune(self, X, y, max_epoch=50):
        for epoch in range(1, max_epoch + 1):
            loss, corr = self.finetune_epoch(epoch, X, y)
            self.scheduler.step(loss)

    def finetune_epoch(self, epoch, X, y):
        self.model.to(self.device)
        self.model.train()

        dlen = len(X)
        trloss = 0
        y_all, y_pred_all = [], []
        pbar = tqdm(self.mtrloader)

        #for x, acc in zip(X, y):
        self.optimizer.zero_grad()
        y_pred, y_dist = self.forward_finetune(X, labels=y, train=True, matrix=False)
        acc_all = y.to(self.device)
        loss = -self.model.mll(y_dist, acc_all)
        loss.backward()
        self.optimizer.step()

        y = y.cpu().detach().tolist()
        y_pred = y_pred.squeeze().tolist()
        y_all += y
        y_pred_all += y_pred
        # print(y_all)
        # print(y_pred_all)
        pbar.set_description(get_log(
            epoch, loss, y_pred, y, 1, 0))
        trloss += float(loss)

        return trloss / dlen, pearsonr(np.array(y_all),
                                       np.array(y_pred_all))[0]

    def meta_train(self):
        sttime = time.time()
        for epoch in range(1, self.max_epoch + 1):
            self.mtrlog.ep_sttime = time.time()
            loss, corr = self.meta_train_epoch(epoch)
            self.scheduler.step(loss)
            self.mtrlog.print_pred_log(loss, corr, 'train', epoch)
            valoss, vacorr = self.meta_validation(epoch)
            if self.max_corr_dict['corr'] < vacorr or epoch==1:
                self.max_corr_dict['corr'] = vacorr
                self.max_corr_dict['epoch'] = epoch
                self.max_corr_dict['loss'] = valoss
                save_model(epoch, self.model, self.model_path, max_corr=True)

            self.mtrlog.print_pred_log(
                valoss, vacorr, 'valid', max_corr_dict=self.max_corr_dict)

            if epoch % self.save_epoch == 0:
                save_model(epoch, self.model, self.model_path)

        self.mtrlog.save_time_log()
        self.mtrlog.max_corr_log(self.max_corr_dict)

    def meta_train_epoch(self, epoch):
        self.model.to(self.device)
        self.model.train()

        self.mtrloader.dataset.set_mode('train')

        dlen = len(self.mtrloader.dataset)
        trloss = 0
        y_all, y_pred_all = [], []
        pbar = tqdm(self.mtrloader)

        for x, g, acc in pbar:
            self.optimizer.zero_grad()
            y_pred, y_dist = self.forward(x, g, labels=acc, train=True, matrix=False)
            y = acc.to(self.device).double()
            print(y.double())
            print(y_dist)
            loss = -self.model.mll(y_dist, y)
            loss.backward()
            self.optimizer.step()

            y = y.tolist()
            y_pred = y_pred.squeeze().tolist()
            y_all += y
            y_pred_all += y_pred
            pbar.set_description(get_log(
                epoch, loss, y_pred, y, self.acc_std, self.acc_mean))
            trloss += float(loss)

        return trloss / dlen, pearsonr(np.array(y_all),
                                       np.array(y_pred_all))[0]

    def get_topk_idx(self, topk=1):
        # self.mtrloader.dataset.set_mode('train')
        # if self.nasbench201 is None:
        #     self.nasbench201 = torch.load(
        #         os.path.join(self.data_path, 'nasbench201.pt'))
        # z_repr = []
        # g_repr = []
        # acc_repr = []
        # for x, g, acc in tqdm(self.mtrloader):
        #     for jdx, graph in enumerate(g):
        #         str = decode_igraph_to_NAS_BENCH_201_string(graph)
        #         arch_idx = -1
        #         for idx, arch_str in enumerate(self.nasbench201['arch']['str']):
        #             if arch_str == str:
        #                 arch_idx = idx
        #                 break
        #         g_repr.append(arch_idx)
        #         acc_repr.append(acc.detach().cpu().numpy()[jdx])
        g_repr = np.load(file='topk_nb201_idx.npy')
        acc_repr = np.load(file='topk_nb201_acc.npy')
        best = np.argsort(-1 * np.array(acc_repr))[:topk]
        return np.array(g_repr)[best], np.array(acc_repr)[best]

    def meta_validation(self, epoch):
        self.model.to(self.device)
        self.model.eval()

        valoss = 0
        self.mtrloader.dataset.set_mode('valid')
        dlen = len(self.mtrloader.dataset)
        y_all, y_pred_all = [], []
        pbar = tqdm(self.mtrloader)

        with torch.no_grad():
            for x, g, acc in pbar:
                y_pred, y_dist = self.forward(x, g, labels=acc, train=False, matrix=False)
                y = acc.to(self.device)
                loss = -self.model.mll(y_dist, y)

                y = y.tolist()
                y_pred = y_pred.squeeze().tolist()
                y_all += y
                y_pred_all += y_pred
                pbar.set_description(get_log(
                    epoch, loss, y_pred, y, self.acc_std, self.acc_mean, tag='val'))
                valoss += float(loss)
                try:
                    pearson_corr = pearsonr(np.array(y_all), np.array(y_pred_all))[0]
                except Exception as e:
                    pearson_corr = 0

        return valoss / dlen, pearson_corr

    def meta_test(self):
        if self.data_name == 'all':
            for data_name in ['cifar10', 'cifar100', 'mnist', 'svhn', 'aircraft', 'pets']:
                acc = self.meta_test_per_dataset(data_name)
        else:
            acc = self.meta_test_per_dataset(self.data_name)
        return acc

    def meta_test_per_dataset(self, data_name):
        self.nasbench201 = torch.load(
            os.path.join(self.data_path, 'nasbench201.pt'))
        all_arch_str = np.array(self.nasbench201['arch']['str'])
        if 'cifar' in data_name:
            all_arch_acc = np.array(self.nasbench201['test-acc'][data_name])
        self.test_dataset = MetaTestDataset(
            self.data_path, data_name, self.num_sample, self.num_class)

        meta_test_path = os.path.join(
            self.save_path, 'meta_test', data_name, 'best_arch')
        if not os.path.exists(meta_test_path):
            os.makedirs(meta_test_path)
        f_arch_str = open(
            os.path.join(meta_test_path, 'architecture.txt'), 'w')
        save_path = os.path.join(meta_test_path, 'accuracy.txt')
        f = open(save_path, 'w')
        arch_runs = []
        elasped_time = []

        if 'cifar' in data_name:
            N = 100
            trials = 100
            runs = 1
            n_init = 5
            acc_runs = []
        else:
            N = 100
            runs = 1
            trials = 30
            n_init = 5
        avg_acc = []
        print(f'==> select top architectures for {data_name} by meta-predictor...')
        for run in range(1, runs + 1):
            inc, _ = self.get_topk_idx(topk=n_init)
            # np.random.seed(self.args.seed)
            # inc = np.random.choice(np.arange(len(all_arch_str)), size=n_init)
            # print("Arch indexes for seed 555: %s" % inc)
            acc_test = []
            x_support = []
            if 'cifar' in data_name:
                for idx in inc:
                    acc_test.append(all_arch_acc[idx])
                    x_support.append(all_arch_str[idx])
            else:
                print("Data name: %s" % data_name)
                if data_name=='pets':
                    if self.args.seed==333:
                        inc = [12058, 8877, 9997, 9968, 3192]
                        acc_test = [28.38, 22.70, 23.60, 16.04, 24.14]
                    elif self.args.seed==444:
                        inc = [12058, 8877, 9997, 9968, 3192]
                        acc_test = [21.53, 22.43, 22.16, 11.62, 16.67]
                    elif self.args.seed==555:
                        inc = [12058, 8877, 9997, 9968, 3192]
                        acc_test = [27.03, 23.42, 15.59, 13.15, 20.36]
                    for arch_idx in inc:
                        x_support.append(all_arch_str[arch_idx])

                elif 'aircraf' in data_name:
                    if self.args.seed==333:
                        inc = [12058, 8877, 9997, 9968, 3192]
                        acc_test = [21.20, 32.00, 33.17, 37.70, 35.72]
                    elif self.args.seed==444:
                        inc = [12058, 8877, 9997, 9968, 3192]
                        acc_test = [22.29, 29.27, 26.54, 30.41, 24.87]
                    elif self.args.seed==555:
                        inc = [12058, 8877, 9997, 9968, 3192]
                        acc_test = [24.48, 31.55, 29.69, 22.14, 29.99]

                    for arch_idx in inc:
                        x_support.append(all_arch_str[arch_idx])

                elif 'mnis' in data_name:
                    if self.args.seed==333:
                        inc = [12058, 8877, 9997, 9968, 3192]
                        acc_test = [99.62, 99.62, 99.67, 99.68, 99.71]
                    elif self.args.seed==444:
                        inc = [12058, 8877, 9997, 9968, 3192]
                        acc_test = [99.70, 99.46, 99.60, 99.64, 99.57]
                    elif self.args.seed==555:
                        inc = [12058, 8877, 9997, 9968, 3192]
                        acc_test = [99.64, 99.68, 99.19, 55.61, 99.64]

                    for arch_idx in inc:
                        x_support.append(all_arch_str[arch_idx])

                elif 'svh' in data_name:
                    if self.args.seed==333:
                        inc = [12058, 8877, 9997, 9968, 3192]
                        acc_test = [95.89, 96.27, 96.38, 96.40, 96.05]

                    elif self.args.seed==444:
                        inc = [12058, 8877, 9997, 9968, 3192]
                        acc_test = [96.30, 95.26, 96.43, 96.32, 95.89]

                    elif self.args.seed==555:
                        inc = [12058, 8877, 9997, 9968, 3192]
                        acc_test = [96.46, 96.46, 62.99, 57.81, 92.36]

                    # acc_test = [96.07, 96.21, 96.45, 96.27, 95.92]
                    for arch_idx in inc:
                        x_support.append(all_arch_str[arch_idx])

                else:
                    for arch_idx in inc:
                        acc_runs = self.train_single_arch(
                            data_name, self.nasbench201['arch']['str'][arch_idx], meta_test_path)
                        print(f'==> save results in {save_path}')
                        for r, acc in enumerate(acc_runs):
                            msg = f'run {r + 1} {acc:.2f} (%)'
                            f.write(msg + '\n');
                            print(msg)

                        m, h = mean_confidence_interval(acc_runs)
                        msg = f'Avg {m:.2f}+-{h.item():.2f} (%)'
                        f.write(msg + '\n');
                        print(msg)
                        acc_test.append(np.mean(acc_runs))
                        x_support.append(all_arch_str[arch_idx])
            # indexes = inc.tolist()
            # Random init for FSBO
            # inc = np.random.choice(len(archs), size=n_init)
            y_support = acc_test
            support = inc  # np.arange(len(inc)).tolist()
            print(support)
            if len(y_support)> 0:
                print("Best: ", np.amax(y_support))
            traj = []
            for trial in range(trials):
                scores = []
                graphs = [self.nasbench201['arch']['igraph'][i] for i in support]
                with torch.no_grad():
                    x = self.collect_data()
                if len(support) > 1:
                    # with torch.no_grad():
                    z = self.model.get_data_and_graph_repr(x, graphs, matrix=False)
                    support_x, support_y = z, np.array(y_support)  # all_arch_acc[support]
                    support_x = torch.Tensor(support_x).to(self.device) #, self.device)
                    support_y = torch.Tensor(support_y.reshape(-1, )).to(self.device) #, self.device)

                    self.finetune(support_x, support_y)
                # torch.cuda.empty_cache()
                for batch in np.arange(0, len(all_arch_str), step=512):
                    if len(all_arch_str) > batch + 512:
                        sample = all_arch_str[batch:batch + 512]
                        # sample_acc = all_arch_acc[batch:batch + 512]
                        indexes = np.arange(batch, batch + 512)
                    else:
                        sample = all_arch_str[batch:]
                        # sample_acc = all_arch_acc[batch:]
                        indexes = np.arange(batch, len(all_arch_str))
                    if len(support)>0 :
                        archs = np.concatenate((all_arch_str[support], all_arch_str[indexes]))
                    else:
                        archs = all_arch_str[indexes]
                    # y = np.concatenate((all_arch_acc[inc], sample_acc))
                    input = []
                    # self.model.eval()
                    graphs = self.get_items(
                        full_target=self.nasbench201['arch']['igraph'],
                        full_source=self.nasbench201['arch']['str'],
                        source=archs.tolist())
                    z = self.model.get_data_and_graph_repr(x, graphs, matrix=False)
                    if len(support)>0:
                        support_x, support_y = z[:len(support)], np.array(y_support) # all_arch_acc[support]
                    else:
                        support_x = None
                        support_y = None
                    query_x, query_y = z[len(support):], np.zeros(len(indexes)) # all_arch_acc[indexes]
                    if support_x is not None:
                        support_x = torch.Tensor(support_x)# , self.device)
                        support_y = torch.Tensor(support_y.reshape(-1, ))# , self.device)
                    query_x = torch.Tensor(query_x) # , self.device)
                    query_y = torch.Tensor(query_y.reshape(-1, )) # , self.device)
                    mu, std = self.get_mu_and_std(x_support=support_x, y_support=support_y, x_query=query_x, y_query=query_y)
                    # torch.cuda.empty_cache()
                    if len(y_support)>0:
                        incumbent = np.amax(y_support)
                    else:
                        incumbent = 0
                    scores_partial = EI(incumbent=incumbent, mu=mu, stddev=std, support=support, return_score=True)
                    scores.extend(scores_partial.tolist())
                    print("Idx : %s" % batch)
                scores = np.array(scores)
                scores[support] = 0
                if len(support)>0:
                    next = np.argmax(scores)
                    support.append(next)
                    print("Next: %s" % next)
                    if self.train_arch and not 'cifar' in data_name:
                        if not 'cifar' in data_name:
                            acc_runs = self.train_single_arch(
                                data_name, self.nasbench201['arch']['str'][next], meta_test_path)
                        print(f'==> save results in {save_path}')
                        for r, acc in enumerate(acc_runs):
                            msg = f'run {r + 1} {acc:.2f} (%)'
                            f.write(msg + '\n');
                            print(msg)

                        m, h = mean_confidence_interval(acc_runs)
                        msg = f'Avg {m:.2f}+-{h.item():.2f} (%)'
                        f.write(msg + '\n');
                        print(msg)
                        y_support.append(m)
                    else:
                        y_support.append(self.nasbench201['test-acc'][data_name][next])
                    x_support.append(all_arch_str[next])
                    traj.append(np.amax(y_support))
                    print("Trial ", trial, " Incumbent: ", np.amax(y_support))
                    print("Arch idx: %s" % support)
                    print("Acc: %s" % y_support)
                else:
                    best = np.argsort(scores)[-n_init:]
                    support = best.tolist()
                    x_support = [all_arch_str[i] for i in support]
                    y_support = [self.nasbench201['test-acc'][data_name][i] for i in support]
                    traj.append(np.amax(y_support))
                    print("Trial ", trial, " Incumbent: ", np.amax(y_support))

            with open('old_TOP%s_random_FSBO_%s_%s_1.json' % (n_init, data_name, self.args.seed), 'a+') as f:
                json.dump(traj, f)
            print("N.Init ", n_init, " Incumbent: ", np.amax(y_support))
            avg_acc.append(np.amax(y_support))
        return np.mean(avg_acc)



    def train_single_arch(self, data_name, arch_str, meta_test_path):
        seeds = (777, 888, 999)
        train_single_model(save_dir=meta_test_path,
                           workers=24,
                           datasets=[data_name],
                           xpaths=[f'{self.data_path}/raw-data/{data_name}'],
                           splits=[0],
                           use_less=False,
                           seeds=seeds,
                           model_str=arch_str,
                           arch_config={'channel': 16, 'num_cells': 5})
        # Changed training time from 49/199
        epoch = 49 if data_name == 'mnist' else 199
        test_acc_lst = []
        for seed in seeds:
            result = torch.load(os.path.join(meta_test_path, f'seed-0{seed}.pth'))
            test_acc_lst.append(result[data_name]['valid_acc1es'][f'x-test@{epoch}'])
        return test_acc_lst

    def select_top_arch_acc(
            self, data_name, y_pred_all, gen_arch_str, N):
        _, sorted_idx = torch.sort(y_pred_all, descending=True)
        gen_test_acc = self.get_items(
            full_target=self.nasbench201['test-acc'][data_name],
            full_source=self.nasbench201['arch']['str'],
            source=gen_arch_str)
        sorted_gen_test_acc = torch.tensor(gen_test_acc)[sorted_idx]
        sotred_gen_arch_str = [gen_arch_str[_] for _ in sorted_idx]

        max_idx = torch.argmax(sorted_gen_test_acc[:N]).item()
        final_acc = sorted_gen_test_acc[:N][max_idx]
        final_str = sotred_gen_arch_str[:N][max_idx]
        return final_acc, final_str

    def select_top_arch(
            self, data_name, y_pred_all, gen_arch_str, N):
        _, sorted_idx = torch.sort(y_pred_all, descending=True)
        sotred_gen_arch_str = [gen_arch_str[_] for _ in sorted_idx]
        final_str = sotred_gen_arch_str[:N]
        return final_str

    def select_top_acc(self, data_name, final_str):
        final_test_acc = self.get_items(
            full_target=self.nasbench201['test-acc'][data_name],
            full_source=self.nasbench201['arch']['str'],
            source=final_str)
        max_test_acc = max(final_test_acc)
        return max_test_acc

    def collect_data(self):
        x_batch = []
        x_batch.append(self.test_dataset[0])
        return torch.stack(x_batch).to(self.device)

    def get_items(self, full_target, full_source, source):
        return [full_target[full_source.index(_)] for _ in source]

    def load_generated_archs(self, data_name, run):
        mtest_path = os.path.join(
            self.save_path, 'meta_test', data_name, 'generated_arch')
        with open(os.path.join(mtest_path, f'run_{run}.txt'), 'r') as f:
            gen_arch_str = [_.split()[0] for _ in f.readlines()[1:]]
        return gen_arch_str
