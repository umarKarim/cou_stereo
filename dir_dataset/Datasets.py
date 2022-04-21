import torch 
import PIL.Image as Image 
import os
import torchvision.transforms as transforms 
import numpy as np 


class KittiCategoryDataset():
    def __init__(self, opts):
        self.opts = opts 
        self.kitti_root = opts.kitti_root 
        self.cat_txt_files_root = opts.cat_txt_files_root 
        self.kitti_eigen_split_file = opts.kitti_eigen_split_file 
        self.cat = opts.cat 
        self.list_of_transforms = [transforms.ToTensor(), transforms.Normalize([0.45, 0.45, 0.45],
                                                                               [0.225, 0.225, 0.225])]
        self.transforms = transforms.Compose(self.list_of_transforms)

        # gathering the category directories 
        txt_files = [self.cat_txt_files_root + x for x in os.listdir(self.cat_txt_files_root) \
            if x.endswith('.txt')]
        txt_file_name = [x for x in txt_files if self.cat in x][0]
        self.seq_names = []
        with open(txt_file_name, 'r') as f:
            for seq in f:
                seq_name = seq.split('\n')[0]
                self.seq_names += [seq_name] 
        self.seq_names = [self.kitti_root + x + '_sync/' for x in self.seq_names]
        # sequences belonging to the current category self.cat 
        self.seq_names = sorted(list(set(self.seq_names)))
        
        directories = [self.kitti_root + x +'/' for x in os.listdir(self.kitti_root) \
            if os.path.isdir(self.kitti_root + x)]
        self.seq_names = [x for x in self.seq_names if x in directories]
        
        self.left_ims = []
        self.right_ims = []
        for seq_name in self.seq_names:
            left_dir = seq_name + 'image_02/data/'
            right_dir = seq_name + 'image_03/data/'
            left_ims = [left_dir + x for x in os.listdir(left_dir) if (x.endswith('.png') \
                and not x.endswith('_d.png') and not x.endswith('l.png')) or x.endswith('.jpg')]
            left_ims = sorted(left_ims)
            right_ims = [right_dir + x for x in os.listdir(right_dir) if (x.endswith('.png') \
                and not x.endswith('_d.png') and not x.endswith('l.png')) or x.endswith('.jpg')]
            right_ims = sorted(right_ims)
            self.left_ims += left_ims 
            self.right_ims += right_ims 
            
        # removing the test images, if they exist :)
        with open(self.kitti_eigen_split_file) as f:
            for s in f:
                seq_name = self.kitti_root + s.split('/')[1].split(' ')[0] + '/'
                frame_name = s.split(' ')[1] + '.png'
                l_name = seq_name + 'image_02/data/' + frame_name 
                r_name = seq_name + 'image_03/data/' + frame_name 
                if l_name in self.left_ims:
                    self.left_ims.remove(l_name)
                if r_name in self.right_ims:
                    self.right_ims.remove(r_name)
        assert len(self.right_ims) == len(self.left_ims), 'Different number of left and right images'

    def __len__(self):
        return len(self.left_ims)

    def __getitem__(self, i):
        left_im = self.transforms(Image.open(self.left_ims[i]))
        right_im = self.transforms(Image.open(self.right_ims[i]))
        flag = self.cat 
        return {'left_im': left_im,
                'right_im': right_im,
                'flag': flag}

    
class KittiPretrainDataset():
    def __init__(self, opts):
        self.opts = opts 
        self.kitti_root = opts.kitti_root 
        self.kitti_pretrain_cats = opts.kitti_pretrain_cats 
        
        self.list_transforms = [transforms.ToTensor(), transforms.Normalize([0.45, 0.45, 0.45],
                                                                            [0.225, 0.225, 0.225])]
        self.transforms = transforms.Compose(self.list_transforms)

        self.left_ims = []
        self.right_ims = []
        self.flags = []
        for pretrain_cat in self.kitti_pretrain_cats:
            self.opts.cat = pretrain_cat 
            CatDataset = KittiCategoryDataset(self.opts)
            self.left_ims += CatDataset.left_ims
            self.right_ims += CatDataset.right_ims
            self.flags += [pretrain_cat] * len(CatDataset.right_ims)
        assert len(self.left_ims) == len(self.right_ims), 'Different number of left and right images'
    
    def __len__(self):
        return len(self.left_ims)

    def __getitem__(self, i):
        left_im = self.transforms(Image.open(self.left_ims[i]))
        right_im = self.transforms(Image.open(self.right_ims[i]))
        flag = self.flags[i]    
        return {'left_im': left_im,
                'right_im': right_im,
                'flag': flag}


class KittiOnlineDataset():
    def __init__(self, opts):
        self.opts = opts 
        self.kitti_root = opts.kitti_root 
        self.kitti_online_cats = opts.kitti_online_cats 

        self.list_transforms = [transforms.ToTensor(), transforms.Normalize([0.45, 0.45, 0.45],
                                                                            [0.225, 0.225, 0.225])]
        self.transforms = transforms.Compose(self.list_transforms)

        self.left_ims = []
        self.right_ims = []
        self.flags = []
        for online_cat in self.kitti_online_cats:
            self.opts.cat = online_cat 
            CatDataset = KittiCategoryDataset(self.opts)
            self.left_ims += CatDataset.left_ims
            self.right_ims += CatDataset.right_ims
            self.flags += [online_cat] * len(CatDataset.right_ims)
        assert len(self.left_ims) == len(self.right_ims), 'Different number of left and right images'
    
    def __len__(self):
        return len(self.left_ims)

    def __getitem__(self, i):
        left_im = self.transforms(Image.open(self.left_ims[i]))
        right_im = self.transforms(Image.open(self.right_ims[i]))
        flag = self.flags[i]    
        return {'left_im': left_im,
                'right_im': right_im,
                'flag': flag}


class KittiTestDataset():
    def __init__(self, opts):
        self.opts = opts 
        self.kitti_test_dir = opts.kitti_test_dir 
        self.kitti_gt_dir = opts.kitti_gt_dir 
        self.kitti_eigen_split_file = opts.kitti_eigen_split_file 
        self.cat_txt_files_root = opts.cat_txt_files_root 
        
        self.list_transforms = [transforms.ToTensor(), transforms.Normalize([0.45, 0.45, 0.45],
                                                                            [0.225, 0.225, 0.225])]
        self.transforms = transforms.Compose(self.list_transforms)
        
        # gathering the left and right images 
        '''left_dir = self.kitti_test_dir + '/color/'
        self.left_ims = [left_dir + x for x in os.listdir(left_dir) if x.endswith('.png') \
            or x.endswith('.jpg')]
        self.left_ims = sorted(self.left_ims)'''
        left_dir = self.kitti_test_dir + 'left/'
        self.left_ims = [left_dir + x for x in os.listdir(left_dir) if x.endswith('.png') \
            or x.endswith('.jpg')]
        self.left_ims = sorted(self.left_ims)
        right_dir = self.kitti_test_dir + 'right/'
        self.right_ims = [right_dir + x for x in os.listdir(right_dir) if x.endswith('.png') \
            or x.endswith('.jpg')]
        self.right_ims = sorted(self.right_ims)

        # gathering the gt ims 
        self.gt_ims = sorted([self.kitti_gt_dir + x for x in os.listdir(self.kitti_gt_dir) if x.endswith('.npy')])
        
        # gathering the flags 
        txt_files = [x for x in os.listdir(self.cat_txt_files_root) if x.endswith('.txt')]
        flag_data = {}
        for txt_file in txt_files:
            category = txt_file.split('.')[0]
            sequences = []
            with open(self.cat_txt_files_root + txt_file, 'r') as f:
                for s in f:
                    seq_name = s.split('\n')[0]
                    sequences.append(seq_name + '_sync')
            flag_data[category] = sequences

        self.flags = []
        with open(self.kitti_eigen_split_file) as f:
            for s in f:
                seq_name = s.split('/')[1].split(' ')[0]
                for cat, seqs in flag_data.items():
                    if seq_name in seqs:
                        self.flags.append(cat)
                        break 
                
        tot_left = len(self.left_ims)
        tot_right = len(self.right_ims)
        tot_gt = len(self.gt_ims)
        tot_flags = len(self.flags)
        assert tot_left == tot_right == tot_gt == tot_flags, 'Right: {}, left: {}, gt: {}, flags: {}'.format(tot_right, tot_left, tot_gt, tot_flags)      
      
    def __len__(self):
        return len(self.left_ims)

    def __getitem__(self, i):
        left_im = self.transforms(Image.open(self.left_ims[i]))
        right_im = self.transforms(Image.open(self.right_ims[i]))
        gt = torch.tensor(np.load(self.gt_ims[i], allow_pickle=True))
        flag = self.flags[i]
        
        return {'left_im': left_im,
                'right_im': right_im,
                'gt': gt}, flag


class VkittiCategoryDataset():
    def __init__(self, opts):
        self.vkitti_root = opts.vkitti_root 
        self.vkitti_gt_dir = opts.vkitti_gt_dir 
        self.cat = opts.cat 
        self.vkitti_train = opts.vkitti_train 
        self.vkitti_test_perc = opts.vkitti_test_perc
        self.gt_frames = opts.gt_frames 

        self.list_transforms = [transforms.ToTensor(), transforms.Normalize([0.45, 0.45, 0.45],
                                                                            [0.225, 0.225, 0.225])]
        self.transforms = transforms.Compose(self.list_transforms)
        
        if self.gt_frames:
            root = self.vkitti_gt_dir 
        else:
            root = self.vkitti_root 
        scenes = [root + x + '/' for x in os.listdir(root) if os.path.isdir(root + x)]
        
        self.left_ims = []
        self.right_ims = []
        for scene in scenes:
            if self.gt_frames:
                left_dir = scene + self.cat + '/frames/depth/Camera_0/'
                right_dir = scene + self.cat + '/frames/depth/Camera_1/'
            else:
                left_dir = scene + self.cat + '/frames/rgb/Camera_0/'
                right_dir = scene + self.cat + '/frames/rgb/Camera_1/'
            left_ims = sorted([left_dir + x for x in os.listdir(left_dir) if x.endswith('.jpg') or \
                x.endswith('.png')])
            right_ims = sorted([right_dir + x for x in os.listdir(right_dir) if x.endswith('.jpg') or \
                x.endswith('.png')])
            limit = int(len(left_ims) * self.vkitti_test_perc)
            # print(len(left_ims), limit)
            if self.vkitti_train:
                left_ims = left_ims[:-limit]
                right_ims = right_ims[:-limit]
            else:
                left_ims = left_ims[-limit:]
                right_ims = right_ims[-limit:]
            self.left_ims += left_ims 
            self.right_ims += right_ims 
        assert len(self.left_ims) == len(self.right_ims), 'Check number of images'

    def __len__(self):
        return len(self.left_ims)

    def __getitem__(self, i):
        left_im = self.transform(Image.open(self.left_ims[i]))
        right_im = self.transform(Image.open(self.right_ims[i]))
        flag = self.cat 
        return {'left_im': left_im,
                'right_im': right_im,
                'flag': flag}


class VkittiPretrainDataset():
    def __init__(self, opts):
        self.opts = opts 
        self.vkitti_pretrain_cats = opts.vkitti_pretrain_cats 
        self.vkitti_test_perc = opts.vkitti_test_perc 
        self.opts.vkitti_train = True 
        self.opts.gt_frames = False 
        
        self.list_transforms = [transforms.ToTensor(), transforms.Normalize([0.45, 0.45, 0.45],
                                                                            [0.225, 0.225, 0.225])]
        self.transforms = transforms.Compose(self.list_transforms)

        self.left_ims = []
        self.right_ims = []
        self.flags = []
        self.gt_ims = []
        for pretrain_cat in self.vkitti_pretrain_cats:
            self.opts.cat = pretrain_cat 
            CatDataset = VkittiCategoryDataset(self.opts)
            self.left_ims += CatDataset.left_ims
            self.right_ims += CatDataset.right_ims
            self.flags += [pretrain_cat] * len(CatDataset.right_ims)
        assert len(self.left_ims) == len(self.right_ims), 'Different number of left and right images'
    
    def __len__(self):
        return len(self.left_ims)

    def __getitem__(self, i):
        left_im = self.transforms(Image.open(self.left_ims[i]))
        right_im = self.transforms(Image.open(self.right_ims[i]))
        flag = self.flags[i]    
        return {'left_im': left_im,
                'right_im': right_im,
                'flag': flag}


class VkittiOnlineDataset():
    def __init__(self, opts):
        self.opts = opts 
        self.vkitti_online_cats = opts.vkitti_online_cats 
        self.opts.vkitti_train = True 
        self.opts.gt_frames = False  
        
        self.list_transforms = [transforms.ToTensor(), transforms.Normalize([0.45, 0.45, 0.45],
                                                                            [0.225, 0.225, 0.225])]
        self.transforms = transforms.Compose(self.list_transforms)

        self.left_ims = []
        self.right_ims = []
        self.flags = []
        for online_cat in self.vkitti_online_cats:
            self.opts.cat = online_cat  
            CatDataset = VkittiCategoryDataset(self.opts)
            self.left_ims += CatDataset.left_ims
            self.right_ims += CatDataset.right_ims
            self.flags += [online_cat] * len(CatDataset.right_ims)
        assert len(self.left_ims) == len(self.right_ims), 'Different number of left and right images'
    
    def __len__(self):
        return len(self.left_ims)

    def __getitem__(self, i):
        left_im = self.transforms(Image.open(self.left_ims[i]))
        right_im = self.transforms(Image.open(self.right_ims[i]))
        flag = self.flags[i]    
        return {'left_im': left_im,
                'right_im': right_im,
                'flag': flag}


class VkittiTestDataset():
    def __init__(self, opts):
        self.opts = opts 
        self.opts.vkitti_train = False 
        self.test_cats = opts.vkitti_pretrain_cats + opts.vkitti_online_cats 
        
        self.list_transforms = [transforms.ToTensor(), transforms.Normalize([0.45, 0.45, 0.45],
                                                                            [0.225, 0.225, 0.225])]
        self.transforms = transforms.Compose(self.list_transforms)

        # gathering the test frames 
        self.left_ims = []
        self.right_ims = []
        self.flags = []
        self.opts.gt_frames = False 
        for cat in self.test_cats:
            self.opts.cat = cat  
            CatDataset = VkittiCategoryDataset(self.opts)
            self.left_ims += CatDataset.left_ims
            self.right_ims += CatDataset.right_ims
            self.flags += [cat] * len(CatDataset.right_ims)
        # gathering the gt frames 
        self.left_gt = []
        self.opts.gt_frames = True 
        for cat in self.test_cats:
            self.opts.cat = cat  
            CatDataset = VkittiCategoryDataset(self.opts)
            self.left_gt += CatDataset.left_ims
            
        tot_left = len(self.left_ims)
        tot_right = len(self.right_ims)
        tot_gt_l = len(self.left_gt)
        
        # assert tot_left == tot_right, 'Different number of left and right images'
        assert tot_left == tot_right == tot_gt_l, 'Check number of left gt frames'
        
    def __len__(self):
        return len(self.left_ims)

    def __getitem__(self, i):
        left_im = self.transforms(Image.open(self.left_ims[i]))
        right_im = self.transforms(Image.open(self.right_ims[i]))
        gt = torch.tensor(np.array(Image.open(self.left_gt[i])))
        flag = self.flags[i]    
        return {'left_im': left_im,
                'right_im': right_im,
                'gt': gt}, flag


class KittiVkittiPretrain():
    def __init__(self, opts):
        self.opts = opts 
        self.frame_size = opts.frame_size 
        self.KittiDataset = KittiPretrainDataset(self.opts)
        self.VkittiDataset = VkittiPretrainDataset(self.opts)
        self.left_ims = self.KittiDataset.left_ims + self.VkittiDataset.left_ims 
        self.right_ims = self.KittiDataset.right_ims + self.VkittiDataset.right_ims 
        self.flags = self.KittiDataset.flags + self.VkittiDataset.flags 

        self.list_transforms = [transforms.ToTensor(), transforms.Normalize([0.45, 0.45, 0.45],
                                                                            [0.225, 0.225, 0.225])]
        self.transforms = transforms.Compose(self.list_transforms)

    def __len__(self):
        return len(self.left_ims)

    def __getitem__(self, i):
        left_im = self.transforms(Image.open(self.left_ims[i]))
        right_im = self.transforms(Image.open(self.right_ims[i]))
        flag = self.flags[i]
        return {'left_im': left_im,
                'right_im': right_im}, flag


class ReplayOnlineDataset():
    def __init__(self, opts, pretrain_opts):
        self.opts = opts 
        self.replay_left_dir = opts.replay_left_dir
        self.replay_right_dir = opts.replay_right_dir
        self.dataset_tag = opts.dataset_tag 
        self.apply_replay = opts.apply_replay
        self.comoda = opts.comoda 
        
        self.list_transforms = [transforms.ToTensor(), transforms.Normalize([0.45, 0.45, 0.45],
                                                                            [0.225, 0.225, 0.225])]
        self.transforms = transforms.Compose(self.list_transforms)

        if self.dataset_tag == 'kitti':
            self.OnlineDataset = KittiOnlineDataset(self.opts)
        else:
            self.OnlineDataset = VkittiOnlineDataset(self.opts)
        self.dataset_len = len(self.OnlineDataset)
        
        if self.apply_replay:
            self.PretrainDataset = KittiVkittiPretrain(pretrain_opts)
            # predefining all the indices and stuff for faster dataloader 
            self.replay_online_toss = np.random.rand(self.dataset_len)
            self.replay_pretrain_toss = np.random.rand(self.dataset_len)
            pretrain_len = len(self.PretrainDataset)
            replay_len = opts.max_replay_frames 
            self.max_pretrain_frames = pretrain_len
            self.pretrain_inds = (np.random.rand(self.dataset_len) * pretrain_len).astype(np.int)
            self.replay_inds = (np.random.rand(self.dataset_len) * replay_len).astype(np.int)
        
    def __len__(self):
        return self.dataset_len 
    
    def __getitem__(self, i):
        flag = self.OnlineDataset.flags[i]
        # choice between replay or online 
        if self.apply_replay:
            if self.replay_online_toss[i] > 0.5 and self.comoda:
                data = self.get_comoda_data(i)
                replay_flag = True
            else:
                # tot_samples = self.max_pretrain_frames + i 
                # online_data_thresh = i / tot_samples 
                online_data_thresh = 0.5
                if self.replay_online_toss[i] > online_data_thresh:
                    data = self.get_replay_data(i)
                    replay_flag = True 
                else:
                    data = self.get_online_data(i)
                    replay_flag = False
        else:
            data = self.get_online_data(i) 
            replay_flag = False
        return data, flag, replay_flag 
    
    def get_online_data(self, i):
        left_im = self.transforms(Image.open(self.OnlineDataset.left_ims[i]))
        right_im = self.transforms(Image.open(self.OnlineDataset.right_ims[i]))
        return {'left_im': left_im,
                'right_im': right_im}

    def get_comoda_data(self, i):
        i_ = self.pretrain_inds[i]
        left_im = self.transforms(Image.open(self.PretrainDataset.left_ims[i_]))
        right_im = self.transforms(Image.open(self.PretrainDataset.right_ims[i_]))
        return {'left_im': left_im,
                'right_im': right_im}
        
    def get_replay_data(self, i):
        rep_l_ims = [self.replay_left_dir + x for x in os.listdir(self.replay_left_dir) if x.endswith('.png') \
            or x.endswith('.jpg')]
        rep_r_ims = [self.replay_right_dir + x for x in os.listdir(self.replay_right_dir) if x.endswith('.png') \
            or x.endswith('.jpg')]
        assert len(rep_l_ims) == len(rep_r_ims), 'Different number of replay in left and right'
        # tot_rep_samples = self.max_pretrain_frames + len(rep_l_ims)
        # pretrain_replay_thresh = self.max_pretrain_frames / tot_rep_samples 
        # choosing between replay or pretrain 
        pretrain_replay_thresh = 0.5
        if self.replay_pretrain_toss[i] > pretrain_replay_thresh and len(rep_l_ims) > 10:
            i_ = self.replay_inds[i] % len(rep_l_ims)
            left_im = self.transforms(Image.open(rep_l_ims[i_]))
            right_im = self.transforms(Image.open(rep_r_ims[i_]))
        else:
            i_ = self.pretrain_inds[i]
            left_im = self.transforms(Image.open(self.PretrainDataset.left_ims[i_]))
            right_im = self.transforms(Image.open(self.PretrainDataset.right_ims[i_]))
        return {'left_im': left_im,
                'right_im': right_im}
                
        
        
