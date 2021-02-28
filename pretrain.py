import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as transforms 
import torchvision.utils as vutils
import torch.utils.data as data 
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os 
import time 

from dir_dataset import Datasets 
from Loss import Loss 
from dir_options.pretrain_options import Options



class PreTrain():
    def __init__(self, opts):
        self.opts = opts 
        self.epochs = opts.epochs 
        self.batch_size = opts.batch_size
        self.shuffle = opts.shuffle 
        self.lr = opts.lr
        self.beta1 = opts.beta1 
        self.beta2 = opts.beta2
        self.console_out = opts.console_out 
        self.save_disp = opts.save_disp 
        self.disp_module = opts.disp_module 
        self.gpus = opts.gpus  
        self.int_result_dir = opts.int_results_dir 
        self.save_model_dir = opts.save_model_dir 
        self.save_model_iter = opts.save_model_iter 
        self.frame_size = opts.frame_size
        self.disp_model_path = opts.disp_model_path
        self.start_time = time.time()
        
        # making the requried directories if not existing 
        if not os.path.exists(self.int_result_dir) and self.int_result_dir is not None:
            os.makedirs(self.int_result_dir)
        if not os.path.exists(self.save_model_dir) and self.save_model_dir is not None:
            os.makedirs(self.save_model_dir)

        # dataloader
        dataset = Datasets.KittiVkittiPretrain(self.opts)
        self.DataLoader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, 
                                          num_workers=8)
        
        # loading the modules 
        if len(self.gpus) == 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:' + str(self.gpus[0]))
        disp_module = importlib.import_module(self.disp_module)
        self.DispModel = disp_module.DispResNet().to(self.device)
        if self.disp_model_path is not None:
            self.DispModel.load_state_dict(torch.load(self.disp_model_path))
        if len(self.gpus) != 0:
            self.DispModel = nn.DataParallel(self.DispModel, self.gpus)
        self.DispModel.to(self.device)
        self.Loss = Loss(opts)

        # the optimizer 
        self.optim = torch.optim.Adam(self.DispModel.parameters(), lr=self.lr, betas=[self.beta1, self.beta2])

        self.start()

    def start(self):
        for epoch in range(self.epochs):
            for i, (in_data, flag) in enumerate(self.DataLoader):
                for key in in_data.keys():
                    in_data[key] = in_data[key].to(self.device)
                    in_data[key] = F.interpolate(in_data[key], self.frame_size, mode='bilinear') 
                out_disp = self.DispModel(in_data['left_im'])
                out_disp = out_disp.squeeze(1)
                losses, net_loss = self.Loss(in_data, out_disp)
                self.optim.zero_grad()
                net_loss.backward()
                self.optim.step() 

                self.console_display(epoch, i, losses)
                int_disp = self.save_int_result(epoch, i, out_disp, in_data)
                self.save_model(epoch, i) 

    def console_display(self, epoch, i, losses):
        if i % self.console_out == 0:
            loss_list = ''
            for key, loss in losses.items():
                loss_list = loss_list + key + ':' + str(loss.item()) + ', ' 
            tt = time.time() - self.start_time
            tot_b = len(self.DataLoader)
            print('Epoch: {}, batch: {} out of {}, time: {}'.format(epoch, i, tot_b, tt))
            print(loss_list)

    def save_int_result(self, epoch, i, out_disp, in_data):
        if i % self.save_disp == 0:
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
    
    def from1ch23ch(self, im):
        assert len(im.size()) == 4
        new_im = im
        if im.size(1) == 1:
            new_im = im / im.max() 
            new_im = torch.cat((new_im, new_im, new_im), dim=1)
        new_im = new_im[0, :, :, :]
        return new_im

    def gray2jet(self, dmap):
        cmap = plt.get_cmap('jet')
        dmap_0 = dmap.cpu().numpy()
        dmap_norm = (dmap_0 - dmap_0.min()) / (dmap_0.max() - dmap_0.min())
        dmap_col = cmap(dmap_norm)
        dmap_col = dmap_col[:, :, 0:3]
        dmap_col = np.transpose(dmap_col, (2, 0, 1))
        return torch.tensor(dmap_col).float().to(self.device)

    def save_model(self, epoch, i):
        global_step = epoch * len(self.DataLoader) + i 
        if global_step % self.save_model_iter == 0:
            ep_str = ('%03d_' % epoch)
            iter_str = ('%05d' % i)
            disp_model_name = self.save_model_dir + 'Disp_' + ep_str + iter_str + '.pth'
            torch.save(self.DispModel.module.state_dict(), disp_model_name)
            print('Model saved')



if __name__ == '__main__':
    Opts = Options()
    OnlineDepth = PreTrain(Opts.opts)
