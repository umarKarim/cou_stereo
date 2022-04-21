import os 
import shutil
import torch 
import numpy as np 
import random

from online_train import OnlineTrain 
from dir_options.online_train_options import Options 



def clear_directories(opts):
    if not os.path.exists(opts.save_model_dir):
        os.makedirs(opts.save_model_dir) 
    else:
        models = [opts.save_model_dir + x for x in os.listdir(opts.save_model_dir) if x.endswith('.pth')]
        [os.remove(x) for x in models]
    if os.path.exists(opts.int_results_dir):
        int_dmaps = [opts.int_results_dir + x for x in os.listdir(opts.int_results_dir) if x.endswith('.png')]
        [os.remove(x) for x in int_dmaps] 
    if os.path.exists(opts.replay_left_dir):
        left_frames = [opts.replay_left_dir + x for x in os.listdir(opts.replay_left_dir) if x.endswith('.png')]
        [os.remove(x) for x in left_frames]
    if os.path.exists(opts.replay_right_dir):
        right_frames = [opts.replay_right_dir + x for x in os.listdir(opts.replay_right_dir) if x.endswith('.png')] 
        [os.remove(x) for x in right_frames]    
    
    
if __name__ == '__main__':


    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    opts = Options().opts 
    
    opts.network = 'sfml'
    opts.disp_model_path = 'trained_models/sfml/pretrained_models/Disp_019_07439.pth'
    # finetuning only
    opts.dataset_tag = 'kitti'
    print('Current dataset: kitti')
    opts.save_model_dir = 'trained_models/{}/online_models_kitti_ft/'.format(opts.network)
    opts.apply_replay = False 
    opts.apply_mem_reg = False 
    clear_directories(opts)
    OnlineTrain(opts)
    
    opts.dataset_tag = 'vkitti'
    print('Current dataset: VKitti')
    opts.save_model_dir = 'trained_models/{}/online_models_vkitti_ft/'.format(opts.network)
    opts.apply_replay = False 
    opts.apply_mem_reg = False 
    clear_directories(opts)
    OnlineTrain(opts)

    # no further training, only pretraining 
    save_model_dir = 'trained_models/{}/online_models_kitti_none/'.format(opts.network)
    disp_names = sorted([x for x in os.listdir('trained_models/{}/online_models_kitti_ft/'.format(opts.network)) 
                        if x.endswith('.pth') and 'Disp' in x])
    pretrained_disp_path = opts.disp_model_path 
    os.makedirs(save_model_dir, exist_ok=True)
    for disp_name in disp_names:
        shutil.copy(pretrained_disp_path, save_model_dir + disp_name)
    
    # no further training, only pretraining 
    save_model_dir = 'trained_models/{}/online_models_vkitti_none/'.format(opts.network)
    disp_names = sorted([x for x in os.listdir('trained_models/{}/online_models_vkitti_ft/'.format(opts.network)) 
                        if x.endswith('.pth') and 'Disp' in x])
    pretrained_disp_path = opts.disp_model_path 
    os.makedirs(save_model_dir, exist_ok=True)
    for disp_name in disp_names:
        shutil.copy(pretrained_disp_path, save_model_dir + disp_name)
    
    
    runs = opts.runs
    for run_id in runs:
        opts.dataset_tag = 'kitti'
        opts.save_model_dir = 'trained_models/{}/online_models_kitti_prop_run'.format(opts.network) + run_id + '/'
        opts.replay_left_dir = 'replay_frames_run' + run_id + '/left/'
        opts.replay_right_dir = 'replay_frames_run' + run_id + '/right/'
        opts.int_results_dir = 'qual_dmaps/int_results_run' + run_id + '/'
        opts.apply_replay = True 
        opts.apply_mem_reg = True 
        clear_directories(opts)
        OnlineTrain(opts)
        
        opts.dataset_tag = 'vkitti'
        opts.save_model_dir = 'trained_models/{}/online_models_vkitti_prop_run'.format(opts.network) + run_id + '/'
        opts.replay_left_dir = 'replay_frames_run' + run_id + '/left/'
        opts.replay_right_dir = 'replay_frames_run' + run_id + '/right/'
        opts.int_results_dir = 'qual_dmaps/int_results' + run_id + '/'
        opts.apply_replay = True 
        opts.apply_mem_reg = True 
        clear_directories(opts)
        OnlineTrain(opts)    
        
        print('Finished {}'.format(run_id))

    for run_id in runs:
        opts.dataset_tag = 'kitti'
        opts.save_model_dir = 'trained_models/{}/online_models_kitti_comoda_run'.format(opts.network) + run_id + '/'
        opts.replay_left_dir = 'replay_frames_run' + run_id + '/left/'
        opts.replay_right_dir = 'replay_frames_run' + run_id + '/right/'
        opts.int_results_dir = 'qual_dmaps/int_results_run' + run_id + '/'
        opts.apply_replay = True 
        opts.apply_mem_reg = False 
        opts.comoda = True 
        clear_directories(opts)
        OnlineTrain(opts)
        
        opts.dataset_tag = 'vkitti'
        opts.save_model_dir = 'trained_models/{}/online_models_vkitti_comoda_run'.format(opts.network) + run_id + '/'
        opts.replay_left_dir = 'replay_frames_run' + run_id + '/left/'
        opts.replay_right_dir = 'replay_frames_run' + run_id + '/right/'
        opts.int_results_dir = 'qual_dmaps/int_results' + run_id + '/'
        opts.apply_replay = True 
        opts.apply_mem_reg = False 
        opts.comoda = True 
        clear_directories(opts)
        OnlineTrain(opts)
        
        print('Finished {}'.format(run_id))
    
    
    
    
    
    
