import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import torchvision 
import importlib 
from dir_options.test_options import Options 
from dir_dataset import Datasets
import torch.utils.data as data 
import matplotlib.pyplot as plt 
import os 
import cv2
import matplotlib.pyplot as plt 
    
    
    
class TestFaster(): 
    def __init__(self, opts, dataloader=None):
        self.opts = opts 
        self.dataset_tag = opts.dataset_tag 
        self.frame_size = opts.frame_size
        self.model_path = opts.model_path 
        self.gpu_id = opts.gpu_id 
        self.disp_module = opts.disp_module
        self.batch_size = opts.batch_size
        self.min_depth = opts.min_depth
        self.qual_results = opts.qual_results 
        self.metrics = opts.metrics
        
        # The data loader 
        # getting the dataloader ready 
        if dataloader == None:
            if self.dataset_tag == 'kitti':
                dataset = Datasets.KittiTestDataset(self.opts)
            elif self.dataset_tag == 'vkitti':
                dataset = Datasets.VkittiTestDataset(self.opts)
            else:
                raise NameError('Dataset not found')
            self.DataLoader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, 
                                            num_workers=16)
        else:
            self.DataLoader = dataloader
            
        if self.dataset_tag == 'kitti':
            self.max_depth = opts.kitti_max_depth
            self.output_dir = opts.kitti_test_output_dir   
        elif self.dataset_tag == 'vkitti':
            self.max_depth = opts.vkitti_max_depth 
            self.output_dir = opts.vkitti_test_output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # loading the model 
        disp_module = importlib.import_module(self.disp_module)
        self.DispNet = disp_module.DispResNet()
        self.DispNet.load_state_dict(torch.load(self.model_path))
        if self.gpu_id is not None:
            self.device = torch.device('cuda:' + str(self.gpu_id[0]))
            self.DispNet = self.DispNet.to(self.device)
            if len(self.gpu_id) > 1:   
                self.DispNet = nn.DataParallel(self.DispNet, self.gpu_id)
        else:
            self.device = torch.device('cpu')
        self.DispNet.eval()
        
    def __call__(self):
        res, flags = self.evaluate()
        net_rmse = np.mean(np.array(res['rmse']))
        net = {}
        cat = {}
        for metric in self.metrics:
            net[metric] = np.mean(res[metric])
            cat[metric] = self.get_catwise_res(res[metric], flags) 
        return net_rmse, net, cat 
    
    def evaluate(self):
        flags = []
        res_dict = {}
        for metric in self.metrics:
            res_dict[metric] = []
            
        with torch.no_grad():
            for i, (batch_data, flag) in enumerate(self.DataLoader):
                flags += flag 
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                batch_data['left_im'] = F.interpolate(batch_data['left_im'], self.frame_size,
                                                         mode='bilinear')
                out_depth = 1.0 / self.DispNet(batch_data['left_im'])
                out_depth = out_depth.squeeze(1)
                gt = batch_data['gt']
                
                if self.qual_results:
                    self.save_result(i, batch_data, out_depth)
                
                out_depth = self.resize_depth(gt, out_depth)
                gt = self.crop_eigen(gt)
                out_depth = self.crop_eigen(out_depth)
                res = self.get_rmse_list(gt, out_depth)
                for metric in self.metrics:
                    res_dict[metric] += res[metric]
        return res_dict, flags 
    
    def resize_depth(self, gt, depth):
        _, gt_h, gt_w = gt.size()
        disp = 1.0 / (depth + 1e-6)
        disp = disp.unsqueeze(1)
        disp = F.interpolate(disp, (gt_h, gt_w), mode='bilinear')
        disp = disp.squeeze(1)
        depth_out = 1 / (disp + 1e-6)
        return depth_out 
    
    def crop_eigen(self, in_im):
        _, h, w = in_im.size()
        min_h = int(0.40810811 * h)
        max_h = int(0.99189189 * h)
        min_w = int(0.03594771 * w)
        max_w = int(0.96405229 * w)
        return in_im[:, min_h: max_h, min_w: max_w]

    def get_catwise_res(self, rmse, flags):
        rmse_dict = {}
        cat_set = list(set(flags))
        rmse = np.array(rmse).astype(np.float)
        flags = np.array(flags)
        for cat in cat_set:
            loc_mask = flags == cat
            cat_rmse = rmse[loc_mask]
            rmse_dict[cat] = np.mean(cat_rmse)
        return rmse_dict 
    
    def get_rmse_list(self, gt, depth):
        b_size = gt.size(0)
        rmse_list = []
        abs_rel_list = []
        sq_rel_list = []
        del_125_list = []
        del_125_2_list = []
        log_rmse_list = []
        for b in range(b_size):
            curr_depth = depth[b, :, :]
            curr_gt = gt[b, :, :]
            mask = (curr_gt > self.min_depth) * (curr_gt < self.max_depth)
            nz_depth = curr_depth[mask]
            nz_gt = curr_gt[mask] 
            depth_med = torch.median(nz_depth)
            gt_med = torch.median(nz_gt)
            scale = gt_med / (1.0 * depth_med) 
            
            rmse = torch.sqrt(((nz_gt - scale * nz_depth) ** 2).mean())
            abs_rel = (torch.abs(nz_gt - scale * nz_depth) / nz_gt).mean() 
            sq_rel = ((nz_gt - scale * nz_depth) ** 2 / nz_gt).mean()
            log_rmse = torch.sqrt(((torch.log(1.0 * scale * nz_depth) - torch.log(1.0 * nz_gt)) ** 2).mean())
            ratio_1 = scale * nz_depth / nz_gt
            ratio_2 = 1.0 / ratio_1
            thresh = torch.max(ratio_1, ratio_2)
            del_125 = ((thresh < 1.25) * 1.0).mean()
            del_125_2 = ((thresh < 1.25**2) * 1.0).mean()
            
            rmse_list.append(rmse.item())
            abs_rel_list.append(abs_rel.item())
            sq_rel_list.append(sq_rel.item())
            log_rmse_list.append(log_rmse.item())
            del_125_list.append(del_125.item())
            del_125_2_list.append(del_125_2.item())
        res = {'rmse': rmse_list,
               'abs_rel': abs_rel_list, 
               'sq_rel': sq_rel_list,
               'log_rmse': log_rmse_list,
               'del_125': del_125_list,
               'del_125_2': del_125_2_list}                
        return res
            
    def save_result(self, i, batch_data, out_depth):
        b = batch_data['left_im'].size(0)
        for ii in range(b):
            curr_im_name = self.output_dir + ('%05d' % i) + '_' + str(ii) + '.png' 
            depth = out_depth[ii, :, :] 
            depth = 1.0 / depth
            depth_3 = self.gray2jet(depth)
            torchvision.utils.save_image(depth_3, curr_im_name)
    
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



if __name__ == '__main__':
    opts = Options().opts 
    TestFaster(opts)


