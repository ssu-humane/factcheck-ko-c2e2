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
- `ss/nei_ss_results.json`

List of the newly processed data.
- `simcse/data/neg_data.csv`  
- `simcse/data/pos_data.csv`  


### C2E2 Pre-Training
1. KoBERT - Supervised SimCSE
    ```
    cd c2e2
    python ./train.py --input_df="neg_data.csv" --max_length=512 --model="kobert_neg" --pos_neg="neg"
    ```
2. KoBERT - Unsupervised SimcCSE
    ```
    cd c2e2
    python ./train.py --input_df="pos_data.csv" --max_length=512 --model="kobert" --pos_neg="pos"
    ```
3. Koelectra - Supervised SimCSE
    ```
    cd c2e2
    python ./train.py --input_df="neg_data.csv" --max_length=512 --model="koelectra_neg" --pos_neg="neg"
    ```
4. Koelectra - Unsupervised SimCSE
    ```
    cd c2e2
    python ./train.py --input_df="pos_data.csv" --max_length=512 --model="kobert_pos" --pos_neg="pos"
    ```
5. KpfBERT - Supervised SimCSE
    ```
    cd c2e2
    python ./train.py --input_df="neg_data.csv" --max_length=512 --model="kpfbert_neg" --pos_neg="neg"
    ```
6. KpfBERT - Unsupervised SimCSE
    ```
    cd c2e2
    python ./train.py --input_df="pos_data.csv" --max_length=512 --model="kpfbert" --pos_neg="pos"
    ```


### Sentence Selection(SS)
1. embedding based similarity
    ```
    python ss/embedding_based_similarity.py --split="test" --gpu_number=0 --checkpoints_dir="./simcse/checkpoints/" --max_length=512 --model="kosimcse_kpfbert_neg" --model_name="kpfbert_neg_checkpoint.pt"
    ```
