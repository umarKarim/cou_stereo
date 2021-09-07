import os 

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
    opts = Options().opts 
    
    # finetuning only
    opts.dataset_tag = 'kitti'
    print('Current dataset: kitti')
    opts.save_model_dir = 'trained_models/online_models_kitti/'
    opts.apply_replay = False 
    opts.apply_mem_reg = False 
    clear_directories(opts)
    OnlineTrain(opts)
    
    opts.dataset_tag = 'vkitti'
    print('Current dataset: VKitti')
    opts.save_model_dir = 'trained_models/online_models_vkitti/'
    opts.apply_replay = False 
    opts.apply_mem_reg = False 
    clear_directories(opts)
    OnlineTrain(opts)
    
    runs = opts.runs
    for run_id in runs:
        opts.dataset_tag = 'kitti'
        opts.save_model_dir = 'trained_models/online_models_kitti_rep_reg_run' + run_id + '/'
        opts.replay_left_dir = 'replay_frames_run' + run_id + '/left/'
        opts.replay_right_dir = 'replay_frames_run' + run_id + '/right/'
        opts.int_results_dir = 'qual_dmaps/int_results_run' + run_id + '/'
        opts.apply_replay = True 
        opts.apply_mem_reg = True 
        clear_directories(opts)
        OnlineTrain(opts)
        
        opts.dataset_tag = 'vkitti'
        opts.save_model_dir = 'trained_models/online_models_vkitti_rep_reg_run' + run_id + '/'
        opts.replay_left_dir = 'replay_frames_run' + run_id + '/left/'
        opts.replay_right_dir = 'replay_frames_run' + run_id + '/right/'
        opts.int_results_dir = 'qual_dmaps/int_results' + run_id + '/'
        opts.apply_replay = True 
        opts.apply_mem_reg = True 
        clear_directories(opts)
        OnlineTrain(opts)    
        
        print('Finished {}'.format(run_id))
    
    
    
    
    
    
