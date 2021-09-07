from evaluation import EvaluateResults 
from dir_options.test_options import Options 
from dir_options.pretrain_options import Options as OptionsPretrain

import numpy as np 
import pandas as pd

def correct_results(results, opts):
    tag = opts.dataset_tag 
    eval_metric = opts.eval_metric
    curr_dist_res = results['curr_dist']
    cross_dist_res = results['cross_dist']
    cross_domain_res = np.mean((results['pretrain_domain'], results['cross_domain']))
    online_res = results['online_adaptation_res']

    res_dict = {'Curr dist': curr_dist_res,
                'Cross_dist': cross_dist_res,
                'Online': online_res,
                'Cross domain': cross_domain_res}
    return res_dict
    
def eval_function(opts):
    if opts.dataset_tag == 'vkitti':
        opts.vkitti_pretrain_categories = ['30-deg-left', '30-deg-right', 'overcast',
        'rain', 'sunset']
        results = EvaluateResults(opts).complete_evaluation()
        return correct_results(results, opts)
        
    elif opts.dataset_tag == 'kitti':
        opts.kitti_pretrain_categories = ['road', 'residential_1']
        results = EvaluateResults(opts).complete_evaluation()
        return correct_results(results, opts)
    else:
        print('Unknown dataset tag')
        return 0

def get_mean_std(res):
    res_mean = {}
    res_std = {}
    for key in res[0].keys():
        curr_res = []
        for i in range(len(res)):
            curr_res.append(res[i][key])
        res_mean[key] = np.mean(curr_res)
        res_std[key] = np.std(curr_res)
    return list(res_mean.values()), list(res_std.values())
    
def get_mean_std_ft(res):
    res_mean = {}
    res_std = {}
    for key in res.keys():
        res_mean[key] = res[key]
        res_std[key] = 0
    return list(res_mean.values()), list(res_std.values())

def display_latex_style_w_std(mean, std):
    h = len(mean)
    w = len(mean[0])
    for r in range(h):
        for c in range(w):
            mean_str = f"{mean[r][c]:0.4f}"
            std_str = f"{std[r][c]:0.4f}"
            complete_string = mean_str + ' + ' + std_str + ' & '
            print(complete_string, end="")
        print('\\\\')

if __name__ == '__main__':
    opts = Options().opts 
    metrics = opts.metrics
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
    
    
    assert len(list_eval_dir_kitti) == \
        len(list_eval_dir_vkitti) == \
                    len(list_results_dir_kitti) == \
                        len(list_results_dir_vkitti), 'Check the number of elements'
    
    for curr_metric in metrics:
        kitti_res = []
        vkitti_res = []
        opts.eval_metric = curr_metric
        for i in range(len(list_eval_dir_kitti)): 
            opts.eval_dir_kitti = list_eval_dir_kitti[i]
            opts.eval_dir_vkitti = list_eval_dir_vkitti[i]
            opts.results_dir_kitti = list_results_dir_kitti[i]
            opts.results_dir_vkitti = list_results_dir_vkitti[i]
            opts.dataset_tag = 'kitti'
            kitti_res.append(eval_function(opts))
            opts.dataset_tag = 'vkitti'
            vkitti_res.append(eval_function(opts))
        
        kitti_ft = kitti_res[0]
        kitti_runs = kitti_res[1:]
        vkitti_ft = vkitti_res[0]
        vkitti_runs = vkitti_res[1:]
        mean_arr = []
        std_arr = []
        mean, std = get_mean_std_ft(kitti_ft)
        mean_arr.append(mean)
        std_arr.append(std)
        mean, std = get_mean_std(kitti_runs)
        mean_arr.append(mean)
        std_arr.append(std)
        mean, std = get_mean_std_ft(vkitti_ft)
        mean_arr.append(mean)
        std_arr.append(std)
        mean, std = get_mean_std(vkitti_runs)
        mean_arr.append(mean)
        std_arr.append(std)
        
        print('Results for {}'.format(curr_metric))
        display_latex_style_w_std(mean_arr, std_arr)
    print('Finished')
