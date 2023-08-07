import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

class ContrastiveDataset(Dataset):
    def __init__(self, tokenizer, original_texts, positive_texts=None, negative_texts=None, max_len=50, pos_neg='pos'):
        self.tokenizer = tokenizer
        self.org = []
        self.pos = []
        self.neg = []
        self.max_len = max_len
        self.pos_neg = pos_neg
        
        if self.pos_neg == 'pos':
            for idx in tqdm(range(len(original_texts))):
                org = original_texts[idx]

                org_input = self.tokenizer(org, padding='max_length', truncation=True,
                                          max_length=self.max_len, return_tensors='pt')
                org_input['input_ids'] = torch.squeeze(org_input['input_ids'])
                org_input['attention_mask'] = torch.squeeze(org_input['attention_mask'])
            
                self.org.append(org_input) 
        else:
            for idx in tqdm(range(len(positive_texts))):
                org = original_texts[idx]
                pos = positive_texts[idx]
                neg = negative_texts[idx]

                org_input = self.tokenizer(org, padding='max_length', truncation=True,
                                          max_length=self.max_len, return_tensors='pt')
                org_input['input_ids'] = torch.squeeze(org_input['input_ids'])
                org_input['attention_mask'] = torch.squeeze(org_input['attention_mask'])

                pos_input = self.tokenizer(pos, padding='max_length', truncation=True,
                                          max_length=self.max_len, return_tensors='pt')
                pos_input['input_ids'] = torch.squeeze(pos_input['input_ids'])
                pos_input['attention_mask'] = torch.squeeze(pos_input['attention_mask'])

                neg_input = self.tokenizer(neg, padding='max_length', truncation=True,
                                          max_length=self.max_len, return_tensors='pt')
                neg_input['input_ids'] = torch.squeeze(neg_input['input_ids'])
                neg_input['attention_mask'] = torch.squeeze(neg_input['attention_mask'])

                self.org.append(org_input)
                self.pos.append(pos_input)
                self.neg.append(neg_input)
                       
            
    def __len__(self):
        return len(self.org)
    
    def __getitem__(self, idx):
        if self.pos_neg == 'pos':
            return self.org[idx] 
        else:
            return self.org[idx], self.pos[idx], self.neg[idx]