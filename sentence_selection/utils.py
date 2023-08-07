import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from kobert_transformers import get_kobert_model
from transformers import ElectraModel, BertModel

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, count=0, best_score=None, patience=3, verbose=True, delta=0, path=''):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = count
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, optimizer, scheduler, args):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, args)
            
        elif score < self.best_score + self.delta:
            print('score:', score)
            print('self.best_score:', self.best_score)
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
                args.early_stop = True
                
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, args)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, scheduler, args):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        torch.save({
                'chunk_num': args.chunk_num,
                'epoch': args.epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_score': self.best_score,
                'early_count': self.counter,
            }, self.path)
        self.val_loss_min = val_loss
        
        
        
        
    
class KoBERT_Encoder(nn.Module):
    def __init__(self, num_cls):
        super(KoBERT_Encoder, self).__init__()
        
        self.dim = 768
        self.encoder = get_kobert_model()
        self.hidden = 100
        self.mlp_projection = nn.Sequential(nn.Linear(self.dim, self.hidden),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden, self.hidden, bias=True))
        
    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids = torch.tensor(input_ids), attention_mask = torch.tensor(attention_mask))
        embedding = output['pooler_output']
        
        return self.mlp_projection(embedding)
    
    
    
    
class KoELECTRA_Encoder(nn.Module):
    def __init__(self, num_cls):
        super(KoELECTRA_Encoder, self).__init__()
        
        self.dim = 768
        self.encoder = ElectraModel.from_pretrained('monologg/koelectra-base-v3-discriminator')
        self.hidden = 100
        self.mlp_projection = nn.Sequential(nn.Linear(self.dim, self.hidden),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden, self.hidden, bias=True))
        
    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
        full_embedding = output['last_hidden_state']
        cls_embedding = full_embedding[:,0,:]
        embedding = cls_embedding
        
        return self.mlp_projection(embedding)
    
    
    
class KPFBERT_Encoder(nn.Module):
    def __init__(self, num_cls):
        super(KPFBERT_Encoder, self).__init__()
        
        self.dim = 768
        self.encoder = BertModel.from_pretrained("jinmang2/kpfbert", add_pooling_layer=False)
        self.hidden = 100
        self.mlp_projection = nn.Sequential(nn.Linear(self.dim, self.hidden),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden, self.hidden, bias=True))
        
    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
        full_embedding = output['last_hidden_state']
        cls_embedding = full_embedding[:,0,:]
        embedding = cls_embedding
        
        return self.mlp_projection(embedding)
    
    