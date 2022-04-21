import torch.utils.data as data 
import os 

from test import TestFaster 
from dir_options.test_options import Options 
from dir_dataset import Datasets 


if __name__ == '__main__':
    opts = Options().opts 
    opts.qual_results = True 
    opts.network = 'sfml'
    
    kitti_online_model_dir = 'trained_models/online_models_kitti/'
    kitti_rep_reg_model_dir = 'trained_models/online_models_kitti_rep_reg_run3/'
    vkitti_online_model_dir = 'trained_models/online_models_vkitti/'
    vkitti_rep_reg_model_dir = 'trained_models/online_models_vkitti_rep_reg_run3/'
    kitti_online_models = sorted([x for x in os.listdir(kitti_online_model_dir) if x.endswith('.pth')])
    kitti_rep_reg_models = sorted([x for x in os.listdir(kitti_rep_reg_model_dir) if x.endswith('.pth')])
    vkitti_online_models = sorted([x for x in os.listdir(vkitti_online_model_dir) if x.endswith('.pth')])
    vkitti_rep_reg_models = sorted([x for x in os.listdir(vkitti_online_model_dir) if x.endswith('.pth')])
    kitti_online_model = kitti_online_model_dir + kitti_online_models[-1]
    kitti_rep_reg_model = kitti_rep_reg_model_dir + kitti_rep_reg_models[-1]
    vkitti_online_model = vkitti_online_model_dir + vkitti_online_models[-1]
    vkitti_rep_reg_model = vkitti_rep_reg_model_dir + vkitti_rep_reg_models[-1]
    model_paths = [kitti_online_model, kitti_rep_reg_model, vkitti_online_model, vkitti_rep_reg_model]
    train_datasets = ['kitti', 'kitti', 'vkitti', 'vkitti']
    output_dirs = ['qual_dmaps/kitti_online_test_res/',
                   'qual_dmaps/kitti_rep_reg_test_res/',
                   'qual_dmaps/vkitti_online_test_res/',
                   'qual_dmaps/vkitti_rep_reg_test_res/']
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)
    assert len(model_paths) == len(train_datasets) == len(output_dirs), 'Different number of models to check entries' 
    
    # the dataloaders
    kitti_dataset = Datasets.KittiTestDataset(opts)
    vkitti_dataset = Datasets.VkittiTestDataset(opts)
    KittiDataLoader = data.DataLoader(kitti_dataset, batch_size=opts.batch_size, shuffle=False, 
                                        num_workers=16, pin_memory=True)
    VkittiDataLoader = data.DataLoader(vkitti_dataset, batch_size=opts.batch_size, shuffle=False, 
                                            num_workers=16, pin_memory=True)

    for model_path, train_dataset, output_dir in zip(model_paths, train_datasets, output_dirs):
        print('====================')
        print('Model path: {}'.format(model_path))
        print('Training Dataset: {}'.format(train_dataset))
        opts.model_path = model_path 
        print('--------------------')
        print('Testing on KITTI')
        output_dir_ = output_dir + 'kitti/'
        opts.kitti_test_output_dir = output_dir_ 
        os.makedirs(opts.kitti_test_output_dir, exist_ok=True)
        opts.dataset_tag = 'kitti'
        TestFaster(opts, KittiDataLoader).__call__()
        print('Results saved at: {}'.format(output_dir_))
        print('---------------------')
        print('Testing on VKITTI')
        output_dir_ = output_dir + 'vkitti/'
        opts.vkitti_test_output_dir = output_dir_ 
        os.makedirs(opts.vkitti_test_output_dir, exist_ok=True)
        opts.dataset_tag = 'vkitti'
        TestFaster(opts, VkittiDataLoader).__call__()
        print('Results saved at: {}'.format(output_dir_))
        print('DONE')

    print('Finished')

