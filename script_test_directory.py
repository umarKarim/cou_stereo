from options.test_options import Options 
from test_directory import EvalDirectory

import time


if __name__ == '__main__':
    opts = Options().opts 
    runs = opts.runs 

    net = 'sfml'
    opts.network = net
    
    list_eval_dir_kitti = [] 
    list_eval_dir_vkitti = []
    list_results_dir_kitti = []
    list_results_dir_vkitti = []

    # finetuning 
    list_eval_dir_kitti.append('trained_models/{}/online_models_kitti_ft/'.format(net))
    list_eval_dir_vkitti.append('trained_models/{}/online_models_vkitti_ft/'.format(net))    
    list_results_dir_kitti.append('results/{}/online_test_loss/kitti_online_ft/'.format(net))    
    list_results_dir_vkitti.append('results/{}/online_test_loss/vkitti_online_ft/'.format(net))

    for run in runs:
        list_eval_dir_kitti.append('trained_models/{}/online_models_kitti_prop_run'.format(net) + run + '/')
        list_eval_dir_vkitti.append('trained_models/{}/online_models_vkitti_prop_run'.format(net) + run + '/')
        list_results_dir_kitti.append('results/{}/prop_test_loss_run'.format(net) + run + '/kitti_online/')
        list_results_dir_vkitti.append('results/{}/prop_test_loss_run'.format(net) + run + '/vkitti_online/')
    
    for i in range(len(list_eval_dir_kitti)): 
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