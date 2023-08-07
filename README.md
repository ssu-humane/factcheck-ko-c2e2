# factcheck-ko-c2e2

This code repository includes the implementation of contrastive learning method for the Korean factcheck dataset.
The implementation is based on the repository for the baseline implementation: https://github.com/hongcheki/factcheck-ko-2021


# Training

### Data
We used the data provided by the baseline.
The data can be obtained through the baseline github.

List of the data pulled by the baseline respository.
- `data/wiki_claims.json`: Human-Annotated Dataset for the Factcheck
- `data/train_val_test_ids.json`: Lists of claim ids for train/validation/test split
- `data/wiki/wiki_docs.json`: Wikipedia documents corresponing to claims in `wiki_claims.json`
- `dr/dr_results.json`

List of the newly processed data.
- `simcse/data/c2e2_data.csv`  
- `simcse/data/simcse_data.csv`  


### C2E2 Pre-Training
1. KPFBERT - C2E2
    ```
    cd simcse
    python ./train.py --input_df="c2e2_data.csv" --max_length=512 --model="kpfbert_c2e2" --pos_neg="c2e2"
    ```
2. KPFBERT - SimCSE
    ```
    cd simcse
    python ./train.py --input_df="simcse_data.csv" --max_length=512 --model="kpfbert_simcse" --pos_neg="simcse"
    ```


### Sentence Selection(SS)
1. embedding based similarity
    ```
    python ss/embedding_based_similarity.py --split="test" --gpu_number=0 --checkpoints_dir="./simcse/checkpoints/" --max_length=512 --model="kosimcse_kpfbert_c2e2" --model_name="kpfbert_c2e2_checkpoint.pt"
    ```
