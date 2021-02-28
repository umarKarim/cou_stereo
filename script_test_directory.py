from dir_options.test_options import Options 
from test_directory import EvalDirectory

import time


if __name__ == '__main__':
    opts = Options().opts 
    runs = opts.runs 
    
    list_eval_dir_kitti = ['trained_models/online_models_kitti/']
    list_eval_dir_vkitti = ['trained_models/online_models_vkitti/']    
    list_results_dir_kitti = ['results/online_test_loss/kitti_online/']    
    list_results_dir_vkitti = ['results/online_test_loss/vkitti_online/']
    
    for run in runs:
        list_eval_dir_kitti.append('trained_models/online_models_kitti_rep_reg_run' + run + '/')
        list_eval_dir_vkitti.append('trained_models/online_models_vkitti_rep_reg_run' + run + '/')
        list_results_dir_kitti.append('results/rep_reg_online_test_loss_run' + run + '/kitti_online/')
        list_results_dir_vkitti.append('results/rep_reg_online_test_loss_run' + run + '/vkitti_online/')
    
    for i in range(len(list_eval_dir_kitti)):
        if i == 0:
            continue 
        st_time = time.time()                       
        opts.eval_dir_kitti = list_eval_dir_kitti[i]
        opts.eval_dir_vkitti = list_eval_dir_vkitti[i]
        opts.results_dir_kitti = list_results_dir_kitti[i]
        opts.results_dir_vkitti = list_results_dir_vkitti[i]
        print('====================================================')
        print('Directory count: {}'.format(i))
        opts.dataset_tag = 'kitti' 
        EvalDirectory(opts)
        opts.dataset_tag = 'vkitti'
        EvalDirectory(opts)
        print('Time taken: {}'.format(time.time() - st_time))
    print('Finished')