import argparse 

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser()

        # dataset related arguments 
        parser.add_argument('--kitti_root', type=str, default='/hdd/local/sdb/umar/codes/datasets/kitti_raw_256/')
        parser.add_argument('--kitti_eigen_split_file', type=str, default='dir_dataset/kitti_eigen_test_split.txt')
        parser.add_argument('--kitti_pretrain_cats', type=list, default=['kitti_road', 
            'kitti_residential_1'])
        parser.add_argument('--cat_txt_files_root', type=str, default='dir_dataset/kitti_cat_classification/')
        parser.add_argument('--vkitti_root', type=str, default='/hdd/local/sdb/umar/codes/datasets/vkitti_rgb_256/')
        parser.add_argument('--vkitti_pretrain_cats', type=list, default=['15-deg-left', '15-deg-right',
        'clone', 'fog', 'morning'])
        parser.add_argument('--vkitti_test_perc', type=float, default=0.1)
        parser.add_argument('--vkitti_gt_dir', type=str, default='/hdd/local/sdb/umar/codes/datasets/vkitti_depth/')
        parser.add_argument('--frame_size', type=list, default=[256, 320])

        # training options 
        parser.add_argument('--epochs', type=int, default=20)
        parser.add_argument('--batch_size', type=int, default=12)
        parser.add_argument('--shuffle', type=bool, default=True)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--beta2', type=float, default=0.999)
        
        # network 
        parser.add_argument('--network', type=str, help='sfml of diffnet', 
                            default='diffnet')
        
        # gpus 
        parser.add_argument('--gpus', type=list, default=[0])
        
        # intermediate results options
        parser.add_argument('--console_out', type=int, default=50)
        parser.add_argument('--save_disp', type=int, default=50)
        parser.add_argument('--int_results_dir', type=str, default='qual_dmaps/int_results/')
        parser.add_argument('--log_tensorboard', type=bool, default=True)
        parser.add_argument('--tboard_dir', type=str, default='tboard_dir/')
        parser.add_argument('--tboard_out', type=int, default=50)

        # disparity module 
        # parser.add_argument('--disp_module', type=str, default='DispResNet')
        
        # model related options 
        parser.add_argument('--save_model_iter', type=int, default=1000)
        parser.add_argument('--save_model_dir', type=str, default='trained_models/diffnet/pretrained_models/')
        parser.add_argument('--disp_model_path', type=str, default=None)

        # loss related options 
        parser.add_argument('--smooth_wt', type=float, default=0.1)
        parser.add_argument('--ssim_wt', type=float, default=0.85)
        parser.add_argument('--l1_wt', type=float, default=0.15)
        parser.add_argument('--ssim_c1', type=float, default=0.01 ** 2)
        parser.add_argument('--ssim_c2', type=float, default=0.03 ** 2)

        self.opts = parser.parse_args()

    def __call__(self):
        return self.opts 
        
