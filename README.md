# 자동화 팩트체킹을 위한 대조학습 방법 (C2E2)

This code repository includes the implementation of contrastive learning method for the Korean factcheck dataset.
The implementation is based on the repository for the baseline implementation: https://github.com/hongcheki/factcheck-ko-2021


## Training

### Data
List of the newly processed data.
- `pretrain/data/c2e2_data.csv`  
- `pretrain/data/simcse_data.csv`  


### C2E2 Pre-Training
1. KPFBERT - C2E2
    ```
    cd pretrain
    python ./train.py --input_df="c2e2_data.csv" --max_length=512 --model="kpfbert_c2e2" --pos_neg="c2e2"
    ```
2. KPFBERT - SimCSE
    ```
    cd pretrain
    python ./train.py --input_df="simcse_data.csv" --max_length=512 --model="kpfbert_simcse" --pos_neg="simcse"
    ```

You can obtain the KPFBERT-C2E2 checkpoint [here](https://drive.google.com/drive/folders/1zGH8MyC1K6tsbSHh24gEUPwXBThWIEmk?usp=sharing)


## Sentence Selection(SS)

### Data
We used the data provided by the baseline.
The data can be obtained through the baseline github.

List of the data pulled by the baseline repository.
- `data/wiki_claims.json`: Human-Annotated Dataset for the Factcheck
- `data/train_val_test_ids.json`: Lists of claim ids for train/validation/test split
- `data/wiki/wiki_docs.json`: Wikipedia documents corresponding to claims in `wiki_claims.json`
- `dr/dr_results.json`


### Example code for sentence selection
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
