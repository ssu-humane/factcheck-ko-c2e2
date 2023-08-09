# 자동화 팩트체킹을 위한 대조학습 방법 (C2E2)

This code repository includes the implementation of a C2E2 contrastive learning method for a Korean factcheck dataset.

## Data

### factcheck-ko
The Korean fact-checking dataset can be obtained from [this repository](https://github.com/hongcheki/factcheck-ko-2021).

- `data/wiki_claims.json`: human-annotated Dataset for the Factcheck
- `data/train_val_test_ids.json`: Lists of claim ids for train/validation/test split
- `data/wiki/wiki_docs.json`: Wikipedia documents corresponding to claims in `wiki_claims.json`
- `dr/dr_results.json`

### Newly processed data

- `pretrain/data/c2e2_data.csv`  
- `pretrain/data/simcse_data.csv`  


## Contrastive pretraining

1. C2E2
    ```
    cd pretrain
    python ./train.py --input_df="c2e2_data.csv" --pos_neg="c2e2"
    ```
2. SimCSE
    ```
    cd pretrain
    python ./train.py --input_df="simcse_data.csv" --pos_neg="simcse"
    ```
- You can obtain the KPFBERT-C2E2 pretrained checkpoint [here](https://drive.google.com/drive/folders/1zGH8MyC1K6tsbSHh24gEUPwXBThWIEmk?usp=sharing).


## Inference (Sentence Selection)

```
python sentence_selection/embedding_based_similarity.py --split="test" --gpu_number=0 --checkpoints_dir="./pretrain/checkpoints/" --max_length=512 --model="kosimcse_kpfbert_c2e2" --model_name="kpfbert_c2e2_checkpoint.pt"
```

## Reference

```bibtex
@article{송선영2023팩트체킹,
  title={자동화 팩트체킹을 위한 대조학습 방법},
  author={송선영 and 안제준 and 박건우},
  journal={정보과학회논문지},
  year={2023}
}
```
