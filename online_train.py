import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.utils as vutils
import torch.utils.data as data 

import time 
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import os 

from dataset.Datasets import ReplayOnlineDataset 
from loss.Loss import Loss
from options.pretrain_options import Options as PretrainOptions  
from regularization import MemRegularizer 
from networks import get_disp_network


class OnlineTrain():
    def __init__(self, opts):
        self.opts = opts  
        
        # training related options 
        self.pretrain_opts = PretrainOptions().opts
        self.batch_size = 1
        self.shuffle = False 
        self.lr = opts.lr
        self.beta1 = opts.beta1 
        self.beta2 = opts.beta2
        self.console_out = opts.console_out 
        self.save_disp = opts.save_disp  
        # self.disp_module = opts.disp_module 
        self.gpus = opts.gpus  
        self.network = opts.network 

        # replay related options 
        self.apply_replay = opts.apply_replay
        self.replay_left_dir = opts.replay_left_dir 
        self.replay_right_dir = opts.replay_right_dir 
        self.train_loss_mean = opts.train_loss_mean 
        self.train_loss_var = opts.train_loss_var
        self.replay_model_lr = opts.replay_model_lr 
        self.replay_model_th = opts.replay_model_th
        self.replay_counter = 0

        # regularization related options 
        self.apply_mem_reg = opts.apply_mem_reg 
        self.mem_reg_wt = opts.mem_reg_wt 
        
        # intermediate results and models related options 
        self.int_result_dir = opts.int_results_dir 
        self.save_model_dir = opts.save_model_dir 
        self.frame_size = opts.frame_size
        self.disp_model_path = opts.disp_model_path
        self.start_time = time.time()
        self.dataset_tag = opts.dataset_tag
        self.model_counter = 0 

        self.prev_flag = 'init'
        self.train_loss_category = []

        if self.int_result_dir is not None and not os.path.exists(self.int_result_dir):
            os.makedirs(self.int_result_dir)
        if self.save_model_dir is not None and not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
        if self.apply_replay and not os.path.exists(self.replay_left_dir):
            os.makedirs(self.replay_left_dir)
        if self.apply_replay and not os.path.exists(self.replay_right_dir):
            os.makedirs(self.replay_right_dir)
        
        # dataloader 
        self.dataset = ReplayOnlineDataset(self.opts, self.pretrain_opts)
        self.DataLoader = data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, 
                                          num_workers=0)

        # loading the modules 
        if len(self.gpus) == 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:' + str(self.gpus[0]))
        self.DispModel = get_disp_network(self.network)
        
        if self.disp_model_path is not None:
            self.DispModel.load_state_dict(torch.load(self.disp_model_path, map_location='cpu'))
        
        if len(self.gpus) != 0:
            self.DispModel = nn.DataParallel(self.DispModel, self.gpus)
        self.Loss = Loss(opts)

        # the optimizer 
        self.optim = torch.optim.Adam(self.DispModel.parameters(), lr=self.lr, betas=[self.beta1, self.beta2])
        
        # memory regularizer 
        if self.apply_mem_reg:
            self.MemReg = MemRegularizer(opts, self.DispModel)

        print('Current dataset: {}'.format(self.dataset_tag))
        print('Total samples: {}'.format(len(self.DataLoader)))
        print('Replay applied: {}'.format(self.apply_replay))  
        print('Regularization applied: {}'.format(self.apply_mem_reg))
        
        self.train()
        
    def train(self):
        epoch = 0
        for i, (in_data, flag, replay_flag) in enumerate(self.DataLoader):
            for key in in_data.keys():
                in_data[key] = in_data[key].to(self.device)
                                
            in_data['left_im'] = F.interpolate(in_data['left_im'], size=self.frame_size, 
                                                    mode='bilinear')
            in_data['right_im'] = F.interpolate(in_data['right_im'], size= self.frame_size, 
                                                    mode='bilinear')
            
            out_disp = self.DispModel(in_data['left_im'], in_data['right_im'])
            out_disp = out_disp.squeeze(1)
            self.optim.zero_grad()
            losses, net_loss = self.Loss(in_data, out_disp)
            if self.apply_mem_reg:
                dist = (net_loss - self.train_loss_mean) ** 2 / self.train_loss_var 
                dist = dist.detach().abs()
                mem_reg_loss = 1e-2 * dist * self.MemReg.mem_regularize_loss(self.DispModel)
            else: 
                mem_reg_loss = torch.tensor(0.0)                
            losses['mem_reg_loss'] = mem_reg_loss
            tot_loss = net_loss + mem_reg_loss 
            
            tot_loss.backward()
            self.optim.step()
            
            if self.apply_mem_reg:
                self.MemReg.update_importance()

            self.console_display(epoch, i, losses)
            self.save_int_result(epoch, i, out_disp, in_data)
            # self.save_model(epoch, i, flag) 
            self.replay_buffer(in_data, net_loss, replay_flag)
            self.update_replay_params(net_loss)
            if self.prev_flag != flag[0]:
                print('Domain changed from {} to {}'.format(self.prev_flag, flag[0]))
            self.prev_flag = flag[0]
        self.save_model(epoch, len(self.DataLoader) - 1, 'abc')
                
    def update_replay_params(self, net_loss):
        net_loss_d = net_loss.detach() 
        sq_diff = (net_loss_d - self.train_loss_mean) ** 2 
        if self.apply_replay or self.apply_mem_reg:
            self.train_loss_mean += self.replay_model_lr * (net_loss_d - self.train_loss_mean)
            self.train_loss_var += self.replay_model_lr * (sq_diff - self.train_loss_var)
                    
    def replay_buffer(self, in_data, net_loss, replay_flag):
        # to save or not to save the new images to the replay directory
        sq_diff = ((net_loss - self.train_loss_mean) ** 2).detach() 
        if (sq_diff > self.train_loss_var) and (not replay_flag) and (self.apply_replay):
            # save samples for replay
            with torch.no_grad():
                left_im = in_data['left_im'][0, :].cpu().numpy()
                right_im = in_data['right_im'][0, :].cpu().numpy()
                left_im = (left_im - left_im.min()) / (left_im.max() - left_im.min())
                right_im = (right_im - right_im.min()) / (right_im.max() - right_im.min())
                im_name = ('%06d.png' % self.replay_counter)
                left_im = np.transpose(left_im, (1, 2, 0))
                right_im = np.transpose(right_im, (1, 2, 0))
                left_im = cv2.cvtColor(left_im, cv2.COLOR_RGB2BGR)
                right_im = cv2.cvtColor(right_im, cv2.COLOR_RGB2BGR)
                cv2.imwrite(self.replay_left_dir + im_name, 255 * left_im)
                cv2.imwrite(self.replay_right_dir + im_name, 255 * right_im)
                self.replay_counter += 1

    def console_display(self, epoch, i, losses):
        if i % self.console_out == 0:
            loss_list = ''
            for key, loss in losses.items():
                loss_list = loss_list + key + ':' + str(loss.item()) + ', ' 
            tt = time.time() - self.start_time
            tot_b = len(self.DataLoader)
            print('Epoch: {}, batch: {} out of {}, time: {}'.format(epoch, i, tot_b, tt))
            print(loss_list)
            print('Estimated mean: {}, estimated variance: {}'.format(self.train_loss_mean, 
                                                                      self.train_loss_var))
            print('------------------------')
            
    def save_int_result(self, epoch, i, out_disp, in_data):
        if i % self.save_disp == 0 and self.int_result_dir is not None:
            with torch.no_grad():
                disp = self.gray2jet(out_disp[0])
                left_im = in_data['left_im'][0]
                right_im = in_data['right_im'][0]
                recon_right_im = self.Loss.recon_im[0]

                hor_im1 = torch.cat((left_im, disp), dim=-1)
                hor_im2 = torch.cat((right_im, recon_right_im), dim=-1)
                comb_im = torch.cat((hor_im1, hor_im2), dim=1)

                ep_str = ('%03d_' % epoch)
                iter_str = ('%05d' % i)
                im_name = self.int_result_dir + ep_str + iter_str + '.png'
                vutils.save_image(comb_im, im_name)
            print('Intermediate result saved')
            return disp

    def gray2jet(self, dmap):
        cmap = plt.get_cmap('magma')
        if len(dmap.size()) == 4:
            dmap_0 = dmap[0, 0, :, :].cpu().numpy()
        elif len(dmap.size()) == 3:
            dmap_0 = dmap[0, :].cpu().numpy()
        elif len(dmap.size()) == 2:
            dmap_0 = dmap.cpu().numpy()
        else:
            raise 'Wrong dimensions of depth: {}'.format(dmap.size())
        dmap_norm = (dmap_0 - dmap_0.min()) / (dmap_0.max() - dmap_0.min())
        dmap_col = cmap(dmap_norm)
        dmap_col = dmap_col[:, :, 0:3]
        dmap_col = np.transpose(dmap_col, (2, 0, 1))
        return torch.tensor(dmap_col).float().to(self.device)

    def save_model(self, epoch, i, flag):
        curr_flag = flag[0]
        # global_step = epoch * len(self.DataLoader) + i 
        if curr_flag != self.prev_flag:
            count_str = ('%02d_' % self.model_counter)
            ep_str = ('%03d_' % epoch)
            iter_str = ('%05d' % i)
            disp_model_name = self.save_model_dir + count_str + self.prev_flag + '_Disp_' + ep_str + iter_str + '.pth'
            torch.save(self.DispModel.module.state_dict(), disp_model_name)
            self.model_counter += 1
            print('Model saved')


