import os
import re
import gc
import csv
import math
import json
import sys
import random
import pickle
import argparse
import logging
import socket
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    ElectraTokenizerFast,
    ElectraForSequenceClassification,
    AutoModel,
    AutoTokenizer,
)

from utils import KoBERT_Encoder, KoELECTRA_Encoder, KPFBERT_Encoder
from sentence_transformers import util
from kobert_tokenizer import KoBERTTokenizer

import warnings
from transformers import logging as trans_logging
warnings.filterwarnings("ignore")
trans_logging.set_verbosity_error()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def cleanse_and_split(text, rm_parens=False):
    while re.search(r"\[[^\.\[\]]*\]", text):
        text = re.sub(r"\[[^\.\[\]]*\]", "", text)
    text = re.sub(r" 이 소리의 정보[^,\)]*\(도움말·정보\)", "", text)
    text = re.sub(r"(\([^\(\)]*)[\s\,]+(듣기|울음소리)[\s\,]+([^\(\)]*\))", r"\g<1>, \g<3>", text)

    if rm_parens:
        while re.search(r"\([^\(\)]*\)", text):
            text = re.sub(r"\([^\(\)]*\)", "", text)
    else:
        while True:
            lang_search = re.search(r"\([^\(\)]*(:)[^\(\)]*\)", text)  # e.g. 영어: / 프랑스어: / 문화어: ... => Delete
            if lang_search:
                lang_start, lang_end = lang_search.span()
                lang_replace = re.sub(r"(?<=[\(\,])[^\,]*:[^\,]*(?=[\,\)])", "", lang_search.group())
                if lang_replace == lang_search.group():
                    logger.warning(f"An unexpected pattern showed up! {text}")
                text = text[:lang_start] + lang_replace + text[lang_end:]
            else:
                break

        text = re.sub("\([\s\,]*\)", "", text)
        while re.search(r"\([^\.\(\)]*\)", text):
            text = re.sub(r"\(([^\.\(\)]*)\)", r" \g<1> ", text)

    text = re.sub(r"·", " ", text)
    text = re.sub(r"<+([^\.<>]*)>+", r"\g<1>", text)
    text = re.sub(r"《+([^\.《》]*)》+", r"\g<1>", text)

    text = re.sub(r"=+.*?=+", "", text)

    text = re.sub(r"[^가-힣a-zA-Z0-9\s\.\?]+", "", text)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"[\n]+", "\n", text)

    test_lst = [t2.strip() for t in text.split("\n") for t2 in re.split(r'[\.\?][\s]|[\.\?]$', t.strip()) if len(t2) >= 10]
    
    return test_lst


def matching_scorer(evidence, candidate):
    evidence_unigrams = evidence.split()
    candidate_unigrams = candidate.split()
    inner_join = (set(evidence_unigrams) & set(candidate_unigrams))
    ev_score = len(inner_join) / len(set(evidence_unigrams))
    cand_score = len(inner_join) / len(set(candidate_unigrams))
    return (ev_score + cand_score) / 2


def labeler(ev, candidates, labels, thres=0.75):
    matched = False
    ev = ev[:-1] if ev[-1] == "." else ev  # Handling double periods (~..)
    for i, cand in enumerate(candidates):
        score = matching_scorer(ev, cand)
        if "." in cand:
            cand_lst = [c for c in cand.split(".") if len(c) >= 10]
            if cand_lst:
                cand_scores = [matching_scorer(ev, c) for c in cand_lst]
                cand_labels = [0] * len(cand_lst)
                cand_max_score = max(cand_scores)
                if (cand_max_score > score) and (cand_max_score > thres):
                    cand_labels[cand_scores.index(max(cand_scores))] = 1
                    del candidates[i], labels[i]
                    candidates.extend(cand_lst)
                    labels.extend(cand_labels)
                    matched = True
                    continue
        if score >= thres:
            labels[candidates.index(cand)] = 1
            matched = True

    return matched, candidates, labels


def get_data(args, split):
    print(f"Make {split} data")

    with open(os.path.join(args.input_dir, "train_val_test_ids.json"), "r") as fp:
        split_ids = json.load(fp)[f"{split}_ids"]

    with open(os.path.join(args.input_dir, "wiki_claims.json"), "r") as fp:
        claims = json.load(fp)
        # nei samples should be dropped because they essentially don't contain gold 'evidence'
        claims = {cid: data for cid, data in claims.items() if cid in split_ids and data["True_False"] != "None"}
        if args.debug:
            claims = {cid: data for i, (cid, data) in enumerate(claims.items()) if i < 500}

    with open(os.path.join(args.dr_dir, "dr_results.json"), "r") as fp:
        dr_results = json.load(fp)
        dr_results = {cid: dr_result for cid, dr_result in dr_results.items() if cid in split_ids}

    with open(os.path.join(args.corpus_dir, "wiki_docs.json"), "r") as fp:
        wiki = json.load(fp)
        wiki_titles = wiki.keys()

    data_list = []
    warnings = defaultdict(dict)  # "id": {"warning message": some info, ...}
    for cid in claims:
        data = claims[cid]
        titles_annotated = list(set([data[f"title{i}"] for i in range(1, 6) if data[f"title{i}"]]))
        if len(titles_annotated) == 0:
            logger.warning(f"claim id {cid} ... No title is annotated. This claim will be Dropped!")
            warnings[cid]["No title"] = []
            continue
        existing_titles = [title for title in list(set(dr_results[cid] + titles_annotated)) if title in wiki_titles]

        candidates = []
        for title in existing_titles:
            documents = wiki[title]
            date = datetime.datetime.strptime(data["Date"], "%Y-%m-%d %H:%M:%S.%f")
            doc_dates = [datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%SZ") for dt in documents.keys()]
            doc_dates = [dt for dt in doc_dates if dt <= date]
            if not doc_dates:
                warnings[cid]["Wrong Date"] = {"Date annotated": data["Date"],
                                               "Dates in downloaded wiki documents": [d.strftime("%Y-%m-%d %H:%M:%S.%f")
                                                                                      for d in doc_dates]}
                continue
            text = documents[max(doc_dates).strftime("%Y-%m-%dT%H:%M:%SZ")]
            text_lst = cleanse_and_split(text)
            candidates.extend(text_lst)

        labels = [0] * len(candidates)

        # ----- Handle evidences ----- #
        evidences = [data[f"evidence{i}"] for i in range(1, 6) if data[f"evidence{i}"]]
        save = True  # Do not save this claim if we can't find the first evidence from candidates
        for i, ev_anno in enumerate(evidences):
            ev_lst = [e for e in cleanse_and_split(ev_anno) if len(e) >= 10]
            # ev = ev_lst[max(range(len(ev_lst)), key=lambda i: len(ev_lst[i]))]  # Take the longest part of the list
            for ev in ev_lst:
                ev_matched, candidates, labels = labeler(ev, candidates, labels)
                if not ev_matched:
                    if "." in ev:  # Handle Special Case: periods in the middle of the evidence sentence
                        matched_bools = []
                        unmatched_lst = []
                        ev2_lst = [e for e in ev.split(".") if len(e) >= 10]
                        for ev2 in ev2_lst:
                            ev2_matched, candidates, labels = labeler(ev2, candidates, labels)
                            matched_bools.append(ev2_matched)
                            if not ev2_matched:
                                unmatched_lst.append(ev2)
                        if sum(matched_bools) > 0:  # any match, append only un-matched ev2
                            if unmatched_lst:
                                logger.warning(
                                    f"claim id {cid} ... Can't find evidence {unmatched_lst} from the candidates.")
                                warnings[cid]["wrong_evidences"] = unmatched_lst
                        else:  # no match
                            logger.warning(f"claim id {cid} ... Can't find evidence [{ev[:20]}] from the candidates.")
                            warnings[cid]["wrong_evidences"] = ev
                    else:
                        logger.warning(f"claim id {cid} ... Can't find evidence [{ev[:20]}] from the candidates.")
                        warnings[cid]["wrong_evidences"] = ev

            if i == 0 and sum(labels) == 0:
                save = False
                logger.warning(f"claim id {cid} ... Can't find the first evidence [{ev_anno[:20]}] from the candidates")
                warnings[cid]["wrong_evidences"] = ev_anno
                break

        if save:
            claim = data['claim']
            data_list.append({
                'id': cid,
                'claim': claim,
                'candidates': candidates,
                'labels': labels,
                'more_than_two': data["more_than_two"]
            })

    total_n_sentences = 0
    total_ratio = 0
    for d in data_list:
        total_n_sentences += len(d["candidates"])
        total_ratio += sum(d["labels"]) / len(d["labels"])

    print(f"<<{split} set ss labelling results>>")
    print(f"# claims that have wrong titles or evidences: {len(warnings)} / {len(claims)}")
    print(f"average # sentences: {total_n_sentences / len(data_list)}")
    print(f"average evidence sentence ratio: {total_ratio / len(data_list)}")
    if not args.debug:
        with open(f"{args.input_dir}/{split}_ss_warnings.json", "w") as fp:
            json.dump(warnings, fp, indent=4, ensure_ascii=False)

    print(f"# claims left: {len(data_list)} / {len(claims)}")
    return data_list




def load_or_make_data_chunks(args, save=True):
    split = args.split
    small = '_small' if args.debug else ''
    data_path = os.path.join(args.temp_dir, f"{split}_data{small}.pickle")
    
    try:      
        with open(data_path, "rb") as fp:
            data = pickle.load(fp)
        
    except FileNotFoundError or EOFError:
        data = get_data(args, split=split)
        
        if save:
            with open(data_path, "wb") as fp:
                pickle.dump(data, fp)

    return data


def convert_bert_features(args, examples, tokenizer, display_examples=False):
    """
    Convert train examples into BERT's input foramt.
    """
    val_features = []
        
    for ex_idx, example in tqdm(enumerate(examples), total=len(examples)):
        sentence_b = tokenizer.tokenize(example['claim'])

        per_claim_features = []
        per_candidate_features = []
        candidate_feats = []
        
        if len(sentence_b) > args.max_length:
            sentence_b = sentence_b[:args.max_length]

        claim_input_ids = tokenizer.convert_tokens_to_ids(sentence_b)
        claim_mask = [1] * len(claim_input_ids)
        
        claim_padding = [0] * (args.max_length - len(claim_input_ids))
        claim_input_ids += claim_padding
        claim_mask += claim_padding
        assert len(claim_input_ids) == args.max_length
        assert len(claim_mask) == args.max_length             
 
        
        for idx in range(len(example['candidates'])):
            cand = example['candidates'][idx]
            label = example['labels'][idx]
            sentence_a = tokenizer.tokenize(cand)
            
            if len(sentence_a) > args.max_length:
                sentence_a = sentence_a[:args.max_length]

            candidate_input_ids = tokenizer.convert_tokens_to_ids(sentence_a)
            candidate_mask = [1] * len(candidate_input_ids)

            # Zero-padding             
            candidate_padding = [0] * (args.max_length - len(candidate_input_ids))
            candidate_input_ids += candidate_padding
            candidate_mask += candidate_padding
            assert len(candidate_input_ids) == args.max_length
            assert len(candidate_mask) == args.max_length            

            if ex_idx < 3 and display_examples:
                print(f"========= Train Example {ex_idx+1} =========")
                print("claim_tokens: %s" % " ".join([str(x) for x in sentence_b]))
                print("claim_input_ids: %s" % " ".join([str(x) for x in claim_input_ids]))
                print("claim_mask: %s" % " ".join([str(x) for x in claim_mask]))
                print()
                print("candidate_tokens: %s" % " ".join([str(x) for x in sentence_a]))
                print("candidate_input_ids: %s" % " ".join([str(x) for x in candidate_input_ids]))
                print("candidate_mask: %s" % " ".join([str(x) for x in candidate_mask]))
                print()
                print("label: %s" % label)
                print("")
                

            claim_feat = {'input_ids': claim_input_ids, 'input_masks': claim_mask, 'label': label}
            candidate_feat = {'input_ids': candidate_input_ids, 'input_masks': candidate_mask, 'label': label}

            candidate_feats.append(candidate_feat)

        per_claim_features.append(claim_feat)
        per_candidate_features.append(candidate_feats)
        candidate_count = len(candidate_feats)
        
        val_features.append((per_claim_features, per_candidate_features, candidate_count, example["id"], example["more_than_two"]))
             
    return val_features



def multi_to_single(args):
    # 대조학습 시 multi gpu를 사용했으나 ss 작업 시 single gpu를 사용
    MODEL_PATH = args.checkpoints_dir + args.model_name
    checkpoint = torch.load(MODEL_PATH, map_location=f"cuda:{args.gpu}")
    for key in list(checkpoint.keys()):
        if 'model.' in key:
            checkpoint[key.replace('model.', '')] = checkpoint[key]
            del checkpoint[key]
        elif 'module.' in key:
            checkpoint[key.replace('module.', '')] = checkpoint[key]
            del checkpoint[key]
        elif 'encoder.' in key:
            checkpoint[key.replace('encoder.', '')] = checkpoint[key]
            del checkpoint[key]

    return checkpoint



def build_ss_model(args, num_labels=2):
    if args.model == "koelectra":
        return ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator",
                                                                cache_dir=args.cache_dir, num_labels=num_labels)
    
    elif args.model == "kosimcse_skt":
        model = AutoModel.from_pretrained('BM-K/KoSimCSE-bert') 
        return model
    
    elif args.model == "kosimcse_kobert_simcse" or args.model == "kosimcse_kobert_c2e2":
        model = KoBERT_Encoder(num_labels)
        checkpoint = multi_to_single(args)
        model.load_state_dict(checkpoint)
        return model

    elif args.model == "kosimcse_koelectra_simcse" or args.model == "kosimcse_koelectra_c2e2":       
        model = KoELECTRA_Encoder(num_labels)
        checkpoint = multi_to_single(args)
        model.load_state_dict(checkpoint)
        return model

    elif args.model == "kosimcse_kpfbert_simcse" or args.model == "kosimcse_kpfbert_c2e2":
        model = KPFBERT_Encoder(num_labels)
        checkpoint = multi_to_single(args)
        model.load_state_dict(checkpoint)
        return model
    
    else:
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',
                                                              cache_dir=args.cache_dir, num_labels=num_labels)
        MODEL_PATH = args.checkpoints_dir + args.model_name
        checkpoint = torch.load(MODEL_PATH, map_location=f"cuda:{args.gpu}")
        model.load_state_dict(checkpoint["state_dict"])
        return model    


    
def main_worker(gpu, features, args):
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)

    model = build_ss_model(args, num_labels=2)
    model = model.to(args.gpu)

    idx_dataset = TensorDataset(torch.LongTensor(range(len(features))))  # number of docs in validation sets
    idx_loader = DataLoader(
        idx_dataset,
        batch_size=1,
        shuffle=False,
    )
    validate(idx_loader, features, model, args)
    return None




    
def calculate_similarity(claim_out, candidate_outs, do_normalize=True):
    if do_normalize:
        claim_out = F.normalize(claim_out, dim=1)
        candidate_outs = F.normalize(candidate_outs, dim=1)

    cos_scores = []    
    for candidate_out in candidate_outs:       
        cos_score = util.pytorch_cos_sim(claim_out, candidate_out)[0]
        cos_scores.append(cos_score.cpu().item())

    return cos_scores
    
    

         
def validate(idx_loader, features, model, args):
    model.eval()
    val_recall_5, val_precision_5 = 0, 0
    pbar = tqdm(total=len(idx_loader), desc="Iteration")

    for idx in idx_loader:
        idx = idx[0].item()
        doc_claim_features, doc_candidate_features, candidate_count, _, more_than_two = features[idx]
        doc_losses, doc_similarity = [], []

        doc_input_ids_claim = torch.LongTensor([x['input_ids'] for x in doc_claim_features])
        doc_input_masks_claim = torch.LongTensor([x['input_masks'] for x in doc_claim_features])

        doc_input_ids_candidate = torch.LongTensor([x['input_ids'] for x in doc_candidate_features[0]])
        doc_input_masks_candidate = torch.LongTensor([x['input_masks'] for x in doc_candidate_features[0]])
        doc_labels = torch.LongTensor([x['label'] for x in doc_candidate_features[0]])
        
        doc_dataset = TensorDataset(doc_input_ids_candidate, doc_input_masks_candidate, doc_labels)
        doc_dataloader = DataLoader(doc_dataset, batch_size=args.val_batchsize, shuffle=False)    
        for batch in doc_dataloader:
            doc_input_ids_candidate, doc_input_masks_candidate, doc_labels_candidate = batch

                    
            with torch.no_grad():
                if args.model == "kosimcse_skt":
                    claim_out = model(
                        input_ids = doc_input_ids_claim.to(args.gpu),
                        attention_mask = doc_input_masks_claim.to(args.gpu),
                    )['pooler_output']
                    candidate_outs = model(
                        input_ids = doc_input_ids_candidate.to(args.gpu),
                        attention_mask = doc_input_masks_candidate.to(args.gpu),
                    )['pooler_output']
                elif args.model == "":  # bert-multilingual
                    claim_out = model(
                        input_ids = doc_input_ids_claim.to(args.gpu),
                        attention_mask = doc_input_masks_claim.to(args.gpu),
                    ).logits
                    candidate_outs = model(
                        input_ids = doc_input_ids_candidate.to(args.gpu),
                        attention_mask = doc_input_masks_candidate.to(args.gpu),
                    ).logits
                else:    
                    claim_out = model(
                        input_ids = doc_input_ids_claim.to(args.gpu),
                        attention_mask = doc_input_masks_claim.to(args.gpu),
                    )
                    candidate_outs = model(
                        input_ids = doc_input_ids_candidate.to(args.gpu),
                        attention_mask = doc_input_masks_candidate.to(args.gpu),
                    )

                cos_scores = calculate_similarity(claim_out, candidate_outs, args)
                doc_similarity.extend(cos_scores)

        doc_similarity = torch.tensor(doc_similarity).to(args.gpu)
        doc_recall_5, doc_precision_5 = calculate_metric(doc_similarity, doc_labels.to(args.gpu), idx)

        val_recall_5 += doc_recall_5
        val_precision_5 += doc_precision_5

        pbar.update(1)    

    val_recall_5 /= len(idx_loader)
    val_precision_5 /= len(idx_loader)

    val_results = torch.tensor([val_recall_5, val_precision_5]).to(args.gpu)

    val_results /= args.n_gpu
    val_recall_5, val_precision_5 = tuple(val_results.tolist())

    # logging validation results
    split_name = "Test"
    logger.info(f"===== {split_name} Done =====")
    print(f'=== {split_name} recall (top5)', val_recall_5)
    print(f'=== {split_name} precision (top5)', val_precision_5)

    results = pd.DataFrame({'recall_5':[val_recall_5],
                            'precision_5':[val_precision_5]})
    results.to_csv(args.checkpoints_dir + './result.csv', sep=',', index=False)
              
                
                
def calculate_metric(doc_similarity, doc_labels, idx):
    true_indices = (doc_labels == 1).nonzero(as_tuple=True)[0]
    
    n_trues = len(true_indices) # 해당 claim의 label 중 1의 갯수
    if n_trues == 0:
        return 0, 0
    
    # recall
    five_or_length = min(5, len(doc_similarity))
    _, topk_indices = doc_similarity.topk(five_or_length)
    n_covered_top_k = sum([1 for idx in topk_indices if idx in true_indices])
    recall_top5 = n_covered_top_k / n_trues

    # precision
    precision_top5 = n_covered_top_k / five_or_length

    return recall_top5, precision_top5
                


def main():
    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument("--input_dir",
                        default="./data",
                        type=str,
                        help="The input data dir.")
    parser.add_argument("--dr_dir",
                        default="./dr",
                        type=str,
                        help="The results of document retrieval dir.")
    parser.add_argument("--corpus_dir",
                        default="./data/wiki/",
                        type=str,
                        help="The wikipedia corpus dir.")
    parser.add_argument("--temp_dir",
                        default="./ss/tmp/",
                        type=str,
                        help="The temp dir where the processed data file will be saved.")
    parser.add_argument("--checkpoints_dir",
                        default="./ss/pretrained_checkpoints/",
                        type=str,
                        help="Where checkpoints will be stored.")
    parser.add_argument("--cache_dir",
                        default="./data/models/",
                        type=str,
                        help="Where do you want to store the pre-trained models"
                        "downloaded from pytorch pretrained model.")
    parser.add_argument("--val_batchsize",
                        default=8,
                        type=int,
                        help="Batch size for validation examples.")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="random seed.")
    parser.add_argument("--max_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after tokenized."
                        "If longer than this, it will be truncated, else will be padded.")
    parser.add_argument('--debug',
                        default=False,
                        action='store_true',
                        help='Use small datasets to debug.')
    parser.add_argument('--model',
                        default="",
                        type=str,
                        help='')
    parser.add_argument('--model_name',
                        default="",
                        type=str,
                        help='pretrained simcse model')
    parser.add_argument('--gpu_number',
                        default=0,
                        type=int,
                        help='')    
    parser.add_argument('--split',
                        default='test',
                        type=str,
                        help='')
    
    args = parser.parse_args()

    args.n_gpu = 1
    args.gpu = args.gpu_number

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG)

    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.model == "koelectra" or args.model == "kosimcse_koelectra_simcse" or args.model == "kosimcse_koelectra_c2e2":
        tokenizer = ElectraTokenizerFast.from_pretrained("monologg/koelectra-base-v3-discriminator")
    elif args.model == "kosimcse_skt":
        tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-bert')
    elif args.model == "kosimcse_kobert_simcse" or args.model == "kosimcse_kobert_c2e2":
        tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    elif args.model == "kosimcse_kpfbert_simcse" or args.model == "kosimcse_kpfbert_c2e2":
        tokenizer = BertTokenizerFast.from_pretrained("jinmang2/kpfbert")
    else: # bert-multilingual
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    data = load_or_make_data_chunks(args)
    features = convert_bert_features(args, data, tokenizer)

    main_worker(args.gpu, features, args)


if __name__ == "__main__":
    print(f"Job is running on {socket.gethostname()}")
    main()
