import torch 
import torch.nn as nn 
from warper import Warper 


class Loss():
    def __init__(self, opts):
        super(Loss, self).__init__()
        self.ssim_wt = opts.ssim_wt 
        self.l1_wt = opts.l1_wt 
        self.smooth_wt = opts.smooth_wt
        self.ssim = SSIM(opts)
        self.warper = Warper(opts)
        # for debugging only
        self.recon_im = []

    def __call__(self, in_data, disp):
        left_im = in_data['left_im']
        right_im = in_data['right_im']
        right_recon = self.warper.warp(left_im, disp)
        self.recon_im = right_recon 
        photo_loss = self.ssim_wt * self.ssim(right_im, right_recon).mean() + \
            self.l1_wt * (right_recon - right_im).abs().mean()
        smooth_loss = self.compute_smooth_loss(disp, left_im)
        net_loss = (1.0 - self.smooth_wt) * photo_loss + self.smooth_wt * smooth_loss 
        return {'photo_loss': photo_loss,
                'smooth_loss': smooth_loss}, net_loss

    def compute_smooth_loss(self, depth, frame):
        depth = depth.unsqueeze(1)
        mean_depth = depth.mean(2, True).mean(3, True) 
        n_depth = depth / (mean_depth + 1e-7) 
        grad_depth_x = torch.abs(n_depth[:, :, :, :-1] - n_depth[:, :, :, 1:])
        grad_depth_y = torch.abs(n_depth[:, :, :-1, :] - n_depth[:, :, 1:, :]) 
        grad_fr_x = torch.mean(torch.abs(frame[:, :, :, :-1] - frame[:, :, :, 1:]), 1, keepdim=True)
        grad_fr_y = torch.mean(torch.abs(frame[:, :, :-1, :] - frame[:, :, 1:, :]), 1, keepdim=True)
        wt_grad_depth_x = grad_depth_x * torch.exp(-grad_fr_x) 
        wt_grad_depth_y = grad_depth_y * torch.exp(-grad_fr_y)
        return torch.mean(wt_grad_depth_x) + torch.mean(wt_grad_depth_y)

        

class SSIM(nn.Module):
    def __init__(self, opts):
        super(SSIM, self).__init__()
        self.c1 = opts.ssim_c1
        self.c2 = opts.ssim_c2 

        self.mean_im1 = nn.AvgPool2d(3, 1)
        self.mean_im2 = nn.AvgPool2d(3, 1)
        self.mean_sig_im1 = nn.AvgPool2d(3, 1)
        self.mean_sig_im2 = nn.AvgPool2d(3, 1)
        self.mean_im1im2 = nn.AvgPool2d(3, 1)
        self.pad = nn.ReflectionPad2d(1)
    
    def forward(self, im1, im2):
        im1 = self.pad(im1)
        im2 = self.pad(im2)
        mean_im1 = self.mean_im1(im1)
        mean_im2 = self.mean_im2(im2)
        sig_im1 = self.mean_sig_im1(im1 * im1) - mean_im1 * mean_im1 
        sig_im2 = self.mean_sig_im2(im2 * im2) - mean_im2 * mean_im2 
        sig_im1_im2 = self.mean_im1im2(im1 * im2) - mean_im1 * mean_im2 
        num = (2 * mean_im1 * mean_im2 + self.c1) * (2 * sig_im1_im2 + self.c2)
        den = (mean_im1 ** 2 + mean_im2 ** 2 + self.c1) * (sig_im1 + sig_im2 + self.c2)
        return ((1 - num / den) /2.0).clamp(0, 1)


if __name__ == '__main__':
    Opts = Options()
    opts = Opts.opts 
    LossClass = Loss(opts)

    b = 10
    nan_losses = 0
    for exp in range(20):

        in_data = {}
        in_data['curr_frame'] = torch.rand(b, 3, 256, 832) 
        in_data['next_frame'] = torch.rand(b, 3, 256, 832)
        in_data['intrinsics'] = torch.rand(3, 3)
        in_data['intrinsics_inv'] = torch.inverse(in_data['intrinsics'])

        out_depth = {}
        out_depth['curr_depth'] = torch.rand(b, 1, 256, 832)
        out_depth['next_depth'] = torch.rand(b, 1, 256, 832)

        out_pose = {}
        out_pose['curr2nxt'] = torch.rand(b, 6) 
        out_pose['curr2nxt_inv'] = torch.rand(b, 6)

        loss_dict, net_loss = LossClass(in_data, out_depth, out_pose)
        print(loss_dict)
        print(net_loss)
        if torch.isnan(net_loss):
            nan_losses += 1
    print(nan_losses)




        

    
    
    