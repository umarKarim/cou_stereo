from evaluate.evaluation import EvaluateResults 
from options.test_options import Options 
import numpy as np 

def correct_results(results, opts):
    curr_dist_res = results['curr_dist']
    cross_dist_res = results['cross_dist']
    
    res_dict = {'Curr dist': curr_dist_res,
                'Cross_dist': cross_dist_res}
    return res_dict
    
def eval_function(opts):
    if opts.dataset_tag == 'vkitti':
        opts.vkitti_pretrain_categories = ['15-deg-left', '15-deg-right',
        'clone', 'fog', 'morning']
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
    net = 'sfml'
    opts.network = net
    
    list_eval_dir_kitti = ['trained_models/{}/online_models_kitti_ft/'.format(net)]
    list_eval_dir_vkitti = ['trained_models/{}/online_models_vkitti_ft/'.format(net)]    
    list_results_dir_kitti = ['results/{}/online_test_loss/kitti_online_ft/'.format(net)]    
    list_results_dir_vkitti = ['results/{}/online_test_loss/vkitti_online_ft/'.format(net)]
    
    for run in runs:
        list_eval_dir_kitti.append('trained_models/{}/online_models_kitti_prop_run'.format(net) + run + '/')
        list_eval_dir_vkitti.append('trained_models/{}/online_models_vkitti_prop_run'.format(net) + run + '/')
        list_results_dir_kitti.append('results/{}/prop_test_loss_run'.format(net) + run + '/kitti_online/')
        list_results_dir_vkitti.append('results/{}/prop_test_loss_run'.format(net) + run + '/vkitti_online/')
    
    assert len(list_eval_dir_kitti) == \
        len(list_eval_dir_vkitti) == \
                    len(list_results_dir_kitti) == \
                        len(list_results_dir_vkitti), 'Check the number of elements'
    # print('Gathered directory names')    
    for curr_metric in metrics:
        kitti_res = []
        vkitti_res = []
        opts.eval_metric = curr_metric
        print(curr_metric)
        for i in range(len(list_eval_dir_kitti)): 
            # print(i)
            opts.eval_dir_kitti = list_eval_dir_kitti[i]
            opts.eval_dir_vkitti = list_eval_dir_vkitti[i]
            opts.results_dir_kitti = list_results_dir_kitti[i]
            opts.results_dir_vkitti = list_results_dir_vkitti[i]
            opts.dataset_tag = 'kitti'
            kitti_res.append(eval_function(opts))
            opts.dataset_tag = 'vkitti'
            vkitti_res.append(eval_function(opts))
        
        # print('Got the results')
        tot_runs = len(runs)
        kitti_ft = kitti_res[0]
        kitti_prop_runs = kitti_res[1:1 + tot_runs]
        vkitti_ft = vkitti_res[0]
        vkitti_prop_runs = vkitti_res[1:1 + tot_runs]
        
        mean_arr = []
        std_arr = []
        mean, std = get_mean_std_ft(kitti_ft)
        mean_arr.append(mean)
        std_arr.append(std)
        mean, std = get_mean_std(kitti_prop_runs)
        mean_arr.append(mean)
        std_arr.append(std)
        
        mean, std = get_mean_std_ft(vkitti_ft)
        mean_arr.append(mean)
        std_arr.append(std)
        mean, std = get_mean_std(vkitti_prop_runs)
        mean_arr.append(mean)
        std_arr.append(std)        
        
        print('Results for {}'.format(curr_metric))
        display_latex_style_w_std(mean_arr, std_arr)
    print('Finished')
