import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, encoder, encoder_name):
        super(Model, self).__init__()
        self.dim = 768
        self.encoder = encoder
        self.encoder_name = encoder_name 
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
