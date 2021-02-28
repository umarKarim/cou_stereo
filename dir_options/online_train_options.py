import argparse


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser()

        # dataset related options 
        parser.add_argument('--dataset_tag', type=str, default='kitti')
        parser.add_argument('--kitti_root', type=str, default='/hdd/local/sdb/umar/kitti_raw_256/')
        parser.add_argument('--vkitti_root', type=str, default='/hdd/local/sdb/umar/vkitti_rgb_256/')
        parser.add_argument('--frame_size', type=str, default='256 320')
        parser.add_argument('--kitti_online_cats', type=list, default=['kitti_residential_2', 'kitti_city', 'kitti_campus'])
        parser.add_argument('--vkitti_online_cats', type=list, default=['30-deg-left', '30-deg-right', 'overcast',
        'rain', 'sunset'])
        parser.add_argument('--cat_txt_files_root', type=str, default='dir_dataset/kitti_cat_classification/')
        parser.add_argument('--vkitti_test_perc', type=float, default=0.1)
        parser.add_argument('--vkitti_gt_dir', type=str, default='/dataset_temp/vkitti_depth/')
        parser.add_argument('--kitti_eigen_split_file', type=str, default='dir_dataset/kitti_eigen_test_split.txt')
        parser.add_argument('--runs', type=list, default=['1', '2', '3'])
        
        # replay related options        
        parser.add_argument('--apply_replay', type=bool, default=True)
        parser.add_argument('--replay_left_dir', type=str, default='replay_frames/left/')
        parser.add_argument('--replay_right_dir', type=str, default='replay_frames/right/')
        parser.add_argument('--train_loss_mean', type=float, default=0.7)
        parser.add_argument('--train_loss_var', type=float, default=0.7)
        parser.add_argument('--replay_model_lr', type=float, default=0.1)
        parser.add_argument('--replay_model_th', type=float, default=1.0)
        parser.add_argument('--max_replay_frames', type=int, default=20000)
                
        # regularization related options 
        parser.add_argument('--apply_mem_reg', type=bool, default=True)
        parser.add_argument('--mem_reg_wt', type=float, default=1.0)
        
        # optimization related options 
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--beta2', type=float, default=0.999)

        # network related options 
        parser.add_argument('--disp_module', type=str, default='DispResNet')
        parser.add_argument('--decoder_in_channels', type=int, default=2048)
        parser.add_argument('--disp_model_path', type=str, default='trained_models/pretrain_models/Disp_019_07439.pth')
        
        # intermediate results realted options 
        parser.add_argument('--console_out', type=int, default=50)
        parser.add_argument('--save_disp', type=int, default=200)
        parser.add_argument('--int_results_dir', type=str, default='qual_dmaps/int_results/')
        
        # saving the model 
        parser.add_argument('--save_model_dir', type=str, default='trained_models/online_models_kitti_replay_reg_adapt/')
        
        # gpus 
        parser.add_argument('--gpus', type=list, default=[0])

        # loss 
        parser.add_argument('--ssim_wt', type=float, default=0.85)
        parser.add_argument('--l1_wt', type=float, default=0.15)
        parser.add_argument('--smooth_wt', type=float, default=0.1)
        parser.add_argument('--ssim_c1', type=float, default=0.01 ** 2)
        parser.add_argument('--ssim_c2', type=float, default=0.03 ** 2)
        
        self.opts = parser.parse_args()
        # changing the frame size 
        frame_size = self.opts.frame_size 
        frame_size = frame_size.split(' ')
        frame_size = [int(x) for x in frame_size]
        self.opts.frame_size = frame_size

    def __call__(self):
        return self.opts

