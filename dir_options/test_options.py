import argparse 

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset_tag', type=str, default='kitti')
        parser.add_argument('--eval_dir_kitti', type=str, default='trained_models/online_models_kitti/')
        parser.add_argument('--eval_dir_vkitti', type=str, default='trained_models/online_models_vkitti/')
        parser.add_argument('--results_dir_kitti', type=str, default='results/online_test_loss/kitti_online/')
        parser.add_argument('--results_dir_vkitti', type=str, default='results/online_test_loss/vkitti_online/')
        parser.add_argument('--metrics', type=list, default=['rmse', 'log_rmse', 'abs_rel', 'sq_rel', 'del_125', 'del_125_2'])
        parser.add_argument('--runs', type=list, default=['1', '2', '3'])    

        parser.add_argument('--qual_results', type=bool, default=False)
        parser.add_argument('--frame_size', type=str, default="256 320")
        parser.add_argument('--shuffle', type=bool, default=False)
        parser.add_argument('--batch_size', type=int, default=1)
        
        parser.add_argument('--disp_module', type=str, default='DispResNet')
        parser.add_argument('--gpu_id', type=int, default=[0])
        
        parser.add_argument('--kitti_max_depth', type=float, default=80.0)
        parser.add_argument('--vkitti_max_depth', type=float, default=8000.0)
        parser.add_argument('--min_depth', type=float, default=1.0e-3)

        parser.add_argument('--kitti_test_output_dir', type=str, default='qual_dmaps/kitti_test_output/')
        parser.add_argument('--vkitti_test_output_dir', type=str, default='qual_dmaps/vkitti_test_output/')
        '''parser.add_argument('--kitti_test_cat_file_name', type=str, default='dir_filenames/kitti_test_cat.npy')
        parser.add_argument('--kitti_eigen_test_split_file', type=str, default='dir_filenames/kitti_eigen_test_split.txt')'''
        
        parser.add_argument('--vkitti_root', type=str, default='/dataset_temp/vkitti_dataset/')
        parser.add_argument('--vkitti_pretrain_cats', type=list, default=['15-deg-left', '15-deg-right',
        'clone', 'fog', 'morning'])
        parser.add_argument('--vkitti_test_perc', type=float, default=0.1)
        parser.add_argument('--kitti_test_dir', type=str, default='/dataset_temp/kitti_mono_test/')
        parser.add_argument('--kitti_gt_dir', type=str, default='/dataset_temp/kitti_test_data/depth/')
        parser.add_argument('--kitti_online_cats', type=list, default=['kitti_residential_2', 'kitti_city'])
        parser.add_argument('--vkitti_gt_dir', type=str, default='/dataset_temp/vkitti_depth/')
        parser.add_argument('--vkitti_online_cats', type=list, default=['30-deg-left', '30-deg-right', 'overcast',
        'rain', 'sunset'])
        parser.add_argument('--kitti_eigen_split_file', type=str, default='dir_dataset/kitti_eigen_test_split.txt')
        parser.add_argument('--cat_txt_files_root', type=str, default='/hdd/local/sdb/umar/codes/continual_stereo/dir_dataset/kitti_cat_classification/')
        
        self.opts = parser.parse_args() 
        frame_size = self.opts.frame_size 
        self.opts.frame_size = [int(x) for x in frame_size.split(' ')]
        
    def __call__(self):
        return self.opts 


