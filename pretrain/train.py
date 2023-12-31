import os
import pickle5 as pickle
import random
import numpy as np
import math
import seaborn as sns
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
import csv

from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizerFast

from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from ContrastiveDataset import ContrastiveDataset
from Encoder import Model
from EarlyStop import EarlyStopping
from ContrastiveLoss import Contrastive_Loss


def set_seed(seed): # 모든 seed 설정 
    n_gpu = torch.cuda.device_count()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def create_data_loader(df, tokenizer, max_len, batch_size, num_workers, training_method):
    if training_method == 'simcse':
        cd = ContrastiveDataset(
            tokenizer = tokenizer,
            original_texts=df.sentence.to_numpy(),
            max_len=max_len,
            training_method=training_method
        )
    else:
        cd = ContrastiveDataset(
            tokenizer = tokenizer,
            original_texts=df.claim.to_numpy(),
            positive_texts=df.positive_candidate.to_numpy(),
            negative_texts=df.negative_candidate.to_numpy(),
            max_len=max_len,
            training_method=training_method
        )
        
    return DataLoader(
        cd,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True
    )
    
    
    
def main():
    parser = argparse.ArgumentParser()
    
    # arguments
    parser.add_argument("--input_dir",
                        default="./data/",
                        type=str,
                        help="The input data dir.")
    parser.add_argument("--input_df",
                        default="c2e2_data.csv",
                        type=str,
                        help="The input data.")
    parser.add_argument("--checkpoints_dir",
                        default="./checkpoints/",
                        type=str,
                        help="Where checkpoints will be stored.")
    parser.add_argument("--batchsize",
                        default=4,
                        type=int,
                        help="Batch size for (positive) training examples.")
    parser.add_argument("--learning_rate",
                        default=1e-3,
                        type=float,
                        help="The initial learning rate.")
    parser.add_argument("--max_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after tokenized."
                        )
    parser.add_argument('--training_method',
                        default="c2e2",
                        type=str,
                        help='supervised learning or unsupervised learning')
    parser.add_argument('--epoch',
                        default=50,
                        type=int,
                        help='')
    parser.add_argument('--early_stop',
                        default=False,
                        action='store_true',
                        help='Whether or not it was stopped early during training')
    parser.add_argument('--early_stop_patience',
                        default=4,
                        type=int,
                        help='Early stop patience')
    parser.add_argument('--weight_decay',
                        default=1e-7,
                        type=float,
                        help="")
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help="")
    parser.add_argument('--temperature',
                        default=0.05,
                        type=float,
                        help="")
    parser.add_argument('--seed',
                        default=123,
                        type=int,
                        help="")
        
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
        
    # device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # random seed option
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    
    args.batch_size = args.batch_size * torch.cuda.device_count()
    
    # Contrastive loss function
    loss_func = Contrastive_Loss(args.temperature, args.batch_size, args.training_method)
    train_loss = []

    # model
    tokenizer = BertTokenizerFast.from_pretrained("jinmang2/kpfbert")
    encoder = BertModel.from_pretrained("jinmang2/kpfbert", add_pooling_layer=False)

    model = Model(encoder)
    model = nn.DataParallel(model)
    model = model.to(args.device)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    
    # data load
    df = pd.read_csv(args.input_dir + args.input_df)
    train_pair, test_pair = train_test_split(df, test_size=0.2, random_state=args.seed)
    valid_pair, test_pair = train_test_split(test_pair, test_size=0.5, random_state=args.seed)

    train_data_loader = create_data_loader(train_pair, tokenizer, args.max_length, args.batch_size, args.num_workers, args.training_method)
    valid_data_loader = create_data_loader(valid_pair, tokenizer, args.max_length, args.batch_size, args.num_workers, args.training_method)
    
    # Early_stopping
    early_stopping = EarlyStopping(patience = args.early_stop_patience, path = args.checkpoints_dir + args.model + '_checkpoint.pt' )
    
    # iter로 Early stop할 때 사용
    step = 0
    iteration_criteria = 44176
    
    with open(args.checkpoints_dir + "loss.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["Itr", "Loss", "Type"])

    # Training 
    print('Start Training')
    for epoch in range(args.epochs):
        tbar1 = tqdm(train_data_loader)

        model.train()
        if args.training_method == 'c2e2': # C2E2 Training 
            for t in tbar1:
                step += args.batch_size
                org_input_ids = t[0]['input_ids']
                org_attention_mask = t[0]['attention_mask']

                pos_input_ids = t[1]['input_ids']
                pos_attention_mask = t[1]['attention_mask']

                neg_input_ids = t[2]['input_ids']
                neg_attention_mask = t[2]['attention_mask']

                outputs = model(
                    input_ids = torch.cat([org_input_ids, pos_input_ids, neg_input_ids]).to(args.device), 
                    attention_mask = torch.cat([org_attention_mask, pos_attention_mask, neg_attention_mask]).to(args.device)
                )

                loss = loss_func(outputs)
                train_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                del org_input_ids, pos_input_ids, neg_input_ids, org_attention_mask, pos_attention_mask, neg_attention_mask, outputs, loss

                if step % iteration_criteria == 0:
                    valid_loss = []
                    train_loss_csv = []
                    valid_loss_csv = []

                    # validate
                    ##############검증##################
                    
                    f = open(args.checkpoints_dir + 'loss.csv','a', newline='')
                    with torch.no_grad():
                        for v in valid_data_loader:
                            org_input_ids = v[0]['input_ids']
                            org_attention_mask = v[0]['attention_mask']

                            pos_input_ids = v[1]['input_ids']
                            pos_attention_mask = v[1]['attention_mask']

                            neg_input_ids = v[2]['input_ids']
                            neg_attention_mask = v[2]['attention_mask']

                            outputs = model(
                                input_ids = torch.cat([org_input_ids, pos_input_ids, neg_input_ids]).to(args.device),
                                attention_mask = torch.cat([org_attention_mask, pos_attention_mask, neg_attention_mask]).to(args.device)
                            )


                            loss = loss_func(outputs)
                            valid_loss.append(loss.item())

                            del org_input_ids, pos_input_ids, neg_input_ids, org_attention_mask, pos_attention_mask, neg_attention_mask, outputs, loss

                    avg_valid_loss = sum(valid_loss) / len(valid_loss)

                    valid_loss_csv.append(step)
                    valid_loss_csv.append(avg_valid_loss)
                    valid_loss_csv.append('Valid')

                    train_loss_csv.append(step)
                    train_loss_csv.append(np.average(train_loss))
                    train_loss_csv.append('Train')

                    wr = csv.writer(f)
                    wr.writerow(valid_loss_csv)

                    wr = csv.writer(f)
                    wr.writerow(train_loss_csv)

                    print(str(epoch), 'th epoch, Avg Valid Loss: ', str(avg_valid_loss))


                    ################early stop################################
                    valid_loss = np.average(valid_loss)
                    early_stopping(valid_loss, model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                    f.close()

        else: # SimCSE training
            for t in tbar1:
                step += args.batch_size

                org_input_ids = t[0]['input_ids']
                org_attention_mask = t[0]['attention_mask']

                outputs = model(
                    input_ids = torch.cat([org_input_ids, org_input_ids]).to(args.device), 
                    attention_mask = torch.cat([org_attention_mask, org_attention_mask]).to(args.device)
                )

                loss = loss_func(outputs)
                train_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                del org_input_ids, org_attention_mask, outputs, loss

                if step % iteration_criteria == 0:
                    valid_loss = []
                    train_loss_csv = []
                    valid_loss_csv = []

                    f = open(args.checkpoints_dir + 'loss.csv','a', newline='')
                    with torch.no_grad():
                        for v in valid_data_loader:
                            org_input_ids = v[0]['input_ids']
                            org_attention_mask = v[0]['attention_mask']

                            outputs = model(
                                input_ids = torch.cat([org_input_ids, org_input_ids]).to(args.device), attention_mask = torch.cat([org_attention_mask, org_attention_mask]).to(args.device)
                            )

                            loss = loss_func(outputs)
                            valid_loss.append(loss.item())

                            del org_input_ids, org_attention_mask, outputs, loss

                        avg_valid_loss = sum(valid_loss) / len(valid_loss)                    
                        valid_loss_csv.append(step)
                        valid_loss_csv.append(avg_valid_loss)
                        valid_loss_csv.append('Valid')

                        train_loss_csv.append(step)
                        train_loss_csv.append(np.average(train_loss)) 
                        train_loss_csv.append('Train')

                        wr = csv.writer(f)
                        wr.writerow(valid_loss_csv)

                        wr = csv.writer(f)
                        wr.writerow(train_loss_csv)

                        print(str(epoch), 'th epoch, Avg Valid Loss: ', str(avg_valid_loss))

                    ################################################
                    valid_loss = np.average(valid_loss)
                    early_stopping(valid_loss, model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break


if __name__ == "__main__":
    main()
