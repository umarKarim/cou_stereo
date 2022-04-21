import torch.utils.data as data 
import os 

from test import TestFaster 
from dir_options.test_options import Options 
from dir_dataset import Datasets 

def qual_eval(model_path: str, out_path: str, dataset_tag: str, network: str = 'diffnet')-> None:
    opts = Options().opts 
    opts.model_path = model_path 
    opts.dataset_tag = dataset_tag
    opts.network = network
    opts.qual_results = True
    if dataset_tag == 'kitti':
        # dataset = Datasets.KittiTestDataset(opts)
        opts.kitti_test_output_dir = out_path
        os.makedirs(opts.kitti_test_output_dir, exist_ok=True)
    elif dataset_tag == 'vkitti':
        # dataset = Datasets.VkittiTestDataset(opts)
        opts.vkitti_test_output_dir = out_path
        os.makedirs(opts.vkitti_test_output_dir, exist_ok=True)
    # dataloader = data.DataLoader(dataset, batch_size=opts.batch_size, 
    #                            shuffle=False, num_workers=16, pin_memory=True)
    _, net, _ = TestFaster(opts).__call__()
    print(net)
    print('Check output at {}'.format(out_path))



if __name__ == '__main__':
    model_path = 'trained_models/diffnet/pretrained_models/Disp_019_02467.pth'
    dataset_tag = 'vkitti'
    out_path = 'prop_pretrained_test_vkitti/'
    qual_eval(model_path=model_path, out_path=out_path, dataset_tag=dataset_tag)


