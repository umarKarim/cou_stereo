import torch 
import torch.nn.functional as F 


class Warper():
    def __init__(self, opts):
        self.opts = opts 
        self.frame_size = opts.frame_size 
        self.gpus = opts.gpus 
        if len(self.gpus) == 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:' + str(self.gpus[0]))

    def warp(self, left_image, disp):
        l_sz = left_image.size()[2:]
        d_sz = disp.size()[1:]
        assert l_sz == d_sz, 'Different size of disparity and image, {} vs {}'.format(l_sz, d_sz)
        # making the grid 
        b = left_image.size(0)
        h = left_image.size(2)
        w = left_image.size(3)
        x_vals = torch.linspace(-1, 1, w).to(self.device)
        y_vals = torch.linspace(-1, 1, h).to(self.device)
        meshy, meshx = torch.meshgrid((y_vals, x_vals))
        grid = torch.stack((meshx, meshy), 2)
        grid = grid.unsqueeze(0)
        grid_ = grid.repeat(b, 1, 1, 1)
        grid_[:, :, :, 0] += disp * 2.0
        recon_right = F.grid_sample(left_image, grid_, mode='bilinear', align_corners=False)
        return recon_right 
