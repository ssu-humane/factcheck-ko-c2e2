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

### Document Retrieval

1. Document retrieval\
    We used the baseline code.\
    Baseline result: 84.18% (recall, entire dataset).


### Sentence Selection(SS)

1. Download Wikipedia documents\
    We should download the Wikipedia documents whose titles are retrieved in DR.
    It takes more than 70 hours with 50000 claims. So, we used the document data 'wiki_docs.json' provided by the baseline.


2. SS - Method 1. sentence pair classification
    ```
    chmod +x ./ss_sentence_pair_classification.sh
    ./ss_sentence_pair_classification.sh
    ```

3. SS - Method 2. embedding based similarity
    ```
    chmod +x ./ss_embedding_based_similarity.sh
    ./ss_embedding_based_similarity.sh
    ```

### Recognizing Textual Entailment(RTE)

1. Get NEI SS results\
    NEI claims don't have gold evidences, thus we need to feed the RTE model with results of Sentence Selection for NEI claims. We used the data provided by the baseline.


2. RTE - Method 1. Sentence pair classification
    ```
    chmod +x ./rte_sentence_pair_classification.sh
    ./rte_sentence_pair_classification.sh
    ```

3. RTE - Method 2. Embedding-based classifier
    ```
    chmod +x ./rte_embedding_based_classifier.sh
    ./rte_embedding_based_classifier.sh
    ```

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
