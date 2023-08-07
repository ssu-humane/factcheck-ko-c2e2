import torch.nn as nn
import torch
import torch.nn.functional as F

class Contrastive_Loss(): 
    def __init__(self, temperature, batch_size, pos_neg):
        self.temperature = temperature
        self.batch_size = batch_size
        self.pos_neg = pos_neg
        
    def __call__(self, out, do_normalize=True):
        if self.pos_neg == 'pos':
            if do_normalize:
                out = F.normalize(out, dim=1)
            batch_size = int(out.shape[0]/2)

            if batch_size != self.batch_size:
                bs = batch_size
            else:
                bs = self.batch_size

            # out_1:x, out_2:pos
            out_1, out_2 = out.split(bs, dim=0) # (B,D), (B,D)

            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2*bs, device=sim_matrix.device)).bool()       
            sim_matrix = sim_matrix.masked_select(mask).view(2*bs, -1)

            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

            loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

            return loss   
        
        else:
            if do_normalize:
                out = F.normalize(out, dim=1)
            batch_size = int(out.shape[0]/3)

            if batch_size != self.batch_size:
                bs = batch_size
            else:
                bs = self.batch_size

            out_1, out_2, out_3 = out.split(bs, dim=0) # (B,D), (B,D), (B,D)

            sim_matrix_pos = torch.exp(torch.mm(out_1, out_2.t().contiguous()) / self.temperature)
            sim_matrix_neg = torch.exp(torch.mm(out_1, out_3.t().contiguous()) / self.temperature)

            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)

            loss = (-torch.log(pos_sim / (sim_matrix_pos.sum(dim=-1) + sim_matrix_neg.sum(dim=-1)))).mean()

            return loss
