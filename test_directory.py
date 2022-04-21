from test import TestFaster
import matplotlib.pyplot as plt 
import os 
import numpy as np
import time 
from dir_dataset import Datasets 
import torch.utils.data as data 



class EvalDirectory():
    def __init__(self, opts):
        self.opts = opts
        self.batch_size = opts.batch_size 
        self.dataset_tag = opts.dataset_tag # dataset over which online training was performed 
        if self.dataset_tag == 'kitti':
            self.eval_dir = opts.eval_dir_kitti 
            self.results_dir = opts.results_dir_kitti
        elif self.dataset_tag == 'vkitti':
            self.eval_dir = opts.eval_dir_vkitti 
            self.results_dir = opts.results_dir_vkitti
        os.makedirs(self.results_dir, exist_ok=True)
        self.metrics = opts.metrics 
        self.metrics = ['rmse', 'log_rmse', 'abs_rel', 'sq_rel', 'del_125', 'del_125_2']
        
        # dataloader 
        kitti_dataset = Datasets.KittiTestDataset(self.opts)
        vkitti_dataset = Datasets.VkittiTestDataset(self.opts)
        self.KittiDataLoader = data.DataLoader(kitti_dataset, batch_size=self.batch_size, shuffle=False, 
                                            num_workers=8, pin_memory=True)
        self.VkittiDataLoader = data.DataLoader(vkitti_dataset, batch_size=self.batch_size, shuffle=False, 
                                             num_workers=8, pin_memory=True)
        # the models to test 
        self.model_names = [sorted([self.eval_dir + x for x in os.listdir(self.eval_dir) if 'Disp' in x])[-1]]
        
        results_dict = self.eval_kitti()
        f_name = self.results_dir + self.dataset_tag + 'train_kittitest_online.npy'
        np.save(f_name, results_dict)
        print('Kitti results are saved')

        results_dict = self.eval_vkitti()
        f_name = self.results_dir + self.dataset_tag + 'train_vkittitest_online.npy'
        np.save(f_name, results_dict)
        print('Vkitti results are saved')
        
    def eval_kitti(self):
        res_dict = {}
        for metric in self.metrics:
            res_dict[metric] = []
        for model in self.model_names:
            self.opts.model_path = model 
            self.opts.dataset_tag = 'kitti'
            print('Testing kitti for model: {} out of : {}'.format(model, len(self.model_names)))
            _, net, cat = TestFaster(self.opts, self.KittiDataLoader).__call__()
            for metric in self.metrics:
                metric_dict = {'net': net[metric],
                               'cat': cat[metric]}
                res_dict[metric].append(metric_dict)
        return res_dict
        
    def eval_vkitti(self):
        res_dict = {}
        for metric in self.metrics:
            res_dict[metric] = []
        for model in self.model_names:
            self.opts.model_path = model 
            self.opts.dataset_tag = 'vkitti'
            print('Testing vkitti for model: {} out of: {}'.format(model, len(self.model_names)))
            _, net, cat = TestFaster(self.opts, self.VkittiDataLoader).__call__()
            for metric in self.metrics:
                metric_dict = {'net': net[metric],
                               'cat': cat[metric]}
                res_dict[metric].append(metric_dict)
        return res_dict
        


if __name__ == '__main__':
    opts = Options().opts 
    EvalDirectory(opts)
