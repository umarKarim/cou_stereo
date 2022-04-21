import numpy as np 
from dir_options.test_options import Options 
from dir_options.pretrain_options import Options as OptionsPretrain
import os 
import matplotlib.pyplot as plt 


class EvaluateResults():
    def __init__(self, opts):
        self.dataset_tag = opts.dataset_tag
        self.eval_metric = opts.eval_metric

        if self.dataset_tag == 'kitti':
            self.results_dir = opts.results_dir_kitti
            npy_files = [self.results_dir + x for x in os.listdir(self.results_dir)]
            curr_dist_file = [x for x in npy_files if 'kittitrain_kittitest' in x][0]
            other_dist_file = [x for x in npy_files if 'kittitrain_vkittitest' in x][0]
            self.res_curr_dist = np.load(curr_dist_file, allow_pickle=True).item()[self.eval_metric]
            self.res_other_dist = np.load(other_dist_file, allow_pickle=True).item()[self.eval_metric]
            self.pretrain_categories = opts.kitti_pretrain_categories
            self.eval_dir = opts.eval_dir_kitti
        else:
            self.results_dir = opts.results_dir_vkitti
            npy_files = [self.results_dir + x for x in os.listdir(self.results_dir)]
            curr_dist_file = [x for x in npy_files if 'vkittitrain_vkittitest' in x][0]
            other_dist_file = [x for x in npy_files if 'vkittitrain_kittitest' in x][0]
            self.res_curr_dist = np.load(curr_dist_file, allow_pickle=True).item()[self.eval_metric]
            self.res_other_dist = np.load(other_dist_file, allow_pickle=True).item()[self.eval_metric]
            self.pretrain_categories = opts.vkitti_pretrain_categories
            self.eval_dir = opts.eval_dir_vkitti 

        self.test_categories = list(self.res_curr_dist[0]['cat'].keys())
        self.test_categories = [x.split('kitti_')[-1] for x in self.test_categories]
        self.pretrain_test_categories = [x for x in self.pretrain_categories if x in self.test_categories]
        # remaining online_train_categories 
        self.online_train_categories = self.get_online_train_categories()
        self.domain_eval_categories = [x for x in self.online_train_categories if x in self.test_categories]
        if self.dataset_tag == 'kitti':
            self.test_categories = ['kitti_' + x for x in self.test_categories]
            self.domain_eval_categories = ['kitti_' + x for x in self.domain_eval_categories]
            self.online_train_categories = ['kitti_' + x for x in self.online_train_categories]
            self.pretrain_test_categories = ['kitti_' + x for x in self.pretrain_test_categories]
            
        # getting the evaluation positions
        self.eval_cat_loc = np.in1d(self.online_train_categories, self.domain_eval_categories).nonzero()[0]
        tot_train_cat = len(self.online_train_categories)
        tot_av_cat = len(self.res_curr_dist)
        # assert tot_train_cat == tot_av_cat, \
        #    'Check number of models. Required: {}, Available: {}'.format(tot_train_cat, tot_av_cat)

    def curr_dist_memory(self):
        self.curr_dist_rmse = []
        for model_ind in range(len(self.res_curr_dist)):
            self.curr_dist_rmse.append(self.res_curr_dist[model_ind]['net'])
        return self.curr_dist_rmse 
    
    def cross_dist_memory(self):
        self.cross_dist_rmse = []
        for model_ind in range(len(self.res_other_dist)):
            self.cross_dist_rmse.append(self.res_other_dist[model_ind]['net'])
        return self.cross_dist_rmse 

    def cross_domain_memory(self):
        # getting results for the current categories
        mem_res_domain = []
        for model_ind in range(len(self.res_curr_dist)):
            model_res = self.res_curr_dist[model_ind]['cat']
            if model_ind in self.eval_cat_loc:
                curr_model_rmse = np.array([])
                curr_model_wt = np.array([])
                for i, category in zip(self.eval_cat_loc, self.domain_eval_categories):
                    if i >= model_ind:
                        continue 
                    else:
                        curr_model_rmse = np.append(curr_model_rmse, model_res[category])
                        curr_model_wt = np.append(curr_model_wt, model_ind - i)
                if len(curr_model_rmse) > 0:
                    mem_res_domain += [np.mean(curr_model_rmse)]
        self.cross_domain_rmse = mem_res_domain 
        return self.cross_domain_rmse
        '''mem_res_domain = []
        for model_ind in range(1):
            model_res = self.res_curr_dist[-1]['cat']
            # if model_ind in self.eval_cat_loc:
            curr_model_rmse = []
            for category in self.domain_eval_categories:
                curr_model_rmse += [model_res[category]]
            mem_res_domain += [np.mean(curr_model_rmse)]
        self.cross_domain_rmse = mem_res_domain 
        # print(self.domain_eval_categories)
        return self.cross_domain_rmse  '''      

    def pretrain_memory(self):
        mem_res = []
        for model_ind in range(len(self.res_curr_dist)):
            model_res = self.res_curr_dist[model_ind]['cat']
            if model_ind in self.eval_cat_loc:
                curr_pretrain_rmse = []
                for category in self.pretrain_test_categories:
                    curr_pretrain_rmse += [model_res[category]]
                mem_res += [np.mean(curr_pretrain_rmse)]
        self.pretrain_rmse = mem_res 
        return self.pretrain_rmse
        '''mem_res = []
        for model_ind in range(1):
            model_res = self.res_curr_dist[-1]['cat']
            # if model_ind in self.eval_cat_loc:
            curr_pretrain_rmse = []
            for category in self.pretrain_test_categories:
                curr_pretrain_rmse += [model_res[category]]
            mem_res += [np.mean(curr_pretrain_rmse)]
            # print(self.pretrain_categories)
        self.pretrain_rmse = mem_res 
        return self.pretrain_rmse''' 


    def online_adaptation(self):
        adaptation_res = []
        for model_ind in range(len(self.res_curr_dist)):
            model_res = self.res_curr_dist[model_ind]['cat']
            if model_ind in self.eval_cat_loc:
                cat_to_check = self.online_train_categories[model_ind]
                adaptation_res += [model_res[cat_to_check]]
        self.online_adaptation_rmse = adaptation_res 
        return self.online_adaptation_rmse 

    def complete_evaluation(self):
        curr_dist_res = self.curr_dist_memory()[-1]
        cross_dist_res = self.cross_dist_memory()[-1]
        # online_adaptation_res = np.mean(self.online_adaptation())
        # cross_domain_res = np.mean(self.cross_domain_memory())
        # pretrain_memory_res = np.mean(self.pretrain_memory())
        return {'curr_dist': curr_dist_res,
                'cross_dist': cross_dist_res}
                # 'cross_domain': cross_domain_res,
                # 'pretrain_domain': pretrain_memory_res,
                # 'online_adaptation_res': online_adaptation_res}

    def get_online_train_categories(self):
        model_names = sorted(x for x in os.listdir(self.eval_dir) if x.endswith('.pth'))
        model_names = [x for x in model_names if 'Disp' in x]
        model_tags = [x.split('Disp')[0] for x in model_names]
        if self.dataset_tag == 'vkitti':
            model_tags = [x[3:-1] for x in model_tags]
        elif self.dataset_tag == 'kitti':
            model_tags = [x.replace('kitti_', '') for x in model_tags]
            model_tags = [x[3:-1] for x in model_tags]
        return model_tags


if __name__ == '__main__':
    opts = Options().opts 
    if opts.dataset_tag == 'nyu':
        train_opts = OptionsPretrain().opts 
        train_opts.root = '/hdd/local/sdb/umar/nyu_indoor/rectified_nyu/'
        # getting the pretrain
        nyu_cat_file_name = train_opts.nyu_cat_file_name 
        train_index_end = train_opts.nyu_train_index_end 
        nyu_cat_name_dict = np.load(nyu_cat_file_name, allow_pickle=True).item()
        categories = sorted(nyu_cat_name_dict.keys())
        train_categories = categories[:categories.index(train_index_end)]
        opts.nyu_pretrain_categories = [x[:-1] for x in train_categories] 
        # print(opts.nyu_pretrain_categories)
        Evaluator = EvaluateResults(opts)
        print('NYU results: {}'.format(Evaluator.complete_evaluation()))
        # plotting the results 
        Evaluator.plot_evolution()
    elif opts.dataset_tag == 'kitti':
        opts.kitti_pretrain_categories = ['road', 'residential_1']
        Evaluator = EvaluateResults(opts)
        print('Kitti Results: {}'.format(Evaluator.complete_evaluation()))
        Evaluator.plot_evolution()
    else:
        print('Unknown dataset tag')

