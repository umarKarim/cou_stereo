import torch 
import torch.nn as nn 


class MemRegularizer():
    def __init__(self, opts, depth_model: nn.Module):
        self.opts = opts 
        self.gpus = opts.gpus 
        if len(self.gpus) == 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:' + str(self.gpus[0]))
        curr_depth_params = self.get_flat_parameters(depth_model)
        self.curr_params = curr_depth_params.detach().clone()
        self.diff = torch.zeros_like(self.curr_params)
        self.all_params = torch.zeros_like(self.curr_params)
            
    def mem_regularize_loss(self, depth_model: nn.Module):
        depth_params = self.get_flat_parameters(depth_model)
        self.all_params = depth_params 
        self.diff = (self.all_params - self.curr_params).abs()
        reg_loss = torch.dot(self.all_params.detach().abs(), self.diff)
        return reg_loss 
    
    def update_importance(self):
        self.curr_params = self.all_params.clone().detach()
        
    def get_flat_parameters(self, model: nn.Module):
        flat_params = torch.cat([p.view(-1) for p in model.parameters()]).to(self.device)
        return flat_params 
    
    