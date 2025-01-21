# MA-COIR

## Overview

This repository contains tools and workflows for data preprocessing, label tree building, model training, evaluation,
and prediction for Mention-Agnostic Concept Recognition through an Indexing-Recognition Framework (MA-COIR).

## Data Preprocessing

### CDR Dataset

1. **Download Data**
    - Download the original dataset from [BC5CDR](https://huggingface.co/datasets/bigbio/bc5cdr/tree/main).
    - Place the following files in `data/cdr/ori`:
        - `CDR_TrainingSet.PubTator.txt`
        - `CDR_DevelopmentSet.PubTator.txt`
        - `CDR_TestSet.PubTator.txt`
    - Download the `c2015.bin` and `d2015.bin` files
      from [Mesh 2015 ASCII](https://nlmpubs.nlm.nih.gov/projects/mesh/2015/asciimesh/) and save them to `data/cdr/ori`.

2. **Generate JSON Data**
    - Navigate to `scripts/utils/preprocess`.
    - Run the following command:
      ```bash
      python get_ori_json_data.py --dataset cdr
      ```

3. **Prepare Label Tree**
    - Build the label tree before model training. Refer to the **Label Tree Building** section if you want to create the
      label tree manually.
    - Alternatively, use the pre-built search indexes in `data/cdr/search_index_related`.

4. **Generate Training Data**
    - After obtaining index-related files, run:
      ```bash
      python get_cr_json_data.py --dataset cdr
      ```

### HPO Dataset

1. **Download Data**
    - Download the dataset from [IHP GSC+](https://github.com/lasigeBioTM/IHP/blob/master/GSC%2B.rar).
    - Extract the `Annotations` and `Text` directories to `data/hpo/ori`.
    - Download `hp.json` from [HPO Ontology](https://hpo.jax.org/data/ontology) and save it in `data/hpo/ori`.

2. **Generate JSON Data**
    - Navigate to `scripts/utils/preprocess`.
    - Run:
      ```bash
      python get_ori_json_data.py --dataset hpo
      ```

3. **Prepare Label Tree**
    - Follow the same steps as described for the CDR dataset.

4. **Generate Training Data**
    - After obtaining index-related files, run:
      ```bash
      python get_cr_json_data.py --dataset hpo
      ```

### HOIP Dataset

1. **Download Data**
    - Download the dataset
      from [HOIP Dataset v1](https://github.com/norikinishida/hoip-dataset/blob/main/releases/v1.tar.gz).
    - Place all files into `data/hoip/ori`.
    - Retain `process_concept_id_list.json` and `test_explicit_ori.json` provided in the directory.

2. **Generate JSON Data**
    - Navigate to `scripts/utils/preprocess`.
    - Run:
      ```bash
      python get_ori_json_data.py --dataset hoip
      ```

3. **Prepare Label Tree**
    - Follow the same steps as described for the CDR dataset.

4. **Generate Training Data**
    - After obtaining index-related files, run:
      ```bash
      python get_cr_json_data.py --dataset hoip
      ```

---

## Label Tree Building

### Non-SSID Index Building

- Navigate to `scripts/utils`.
- Run:
  ```bash
  python build_non_ssid_index.py
  ```

### SSID Index Building

- Navigate to `scripts/utils`.
- Run:
  ```bash
  python build_label_tree.py --dataset {dataset}
  ```

- If there is an embedding files of all concepts in the `models/{dataset}/label_tree/`
  like `Entity_NQ_bert_512_k10_c10_seed_7.pkl`, you can set `--read_emb` to `1` to save time:
  ```bash
  python build_label_tree.py --dataset {dataset} --read_emb 1
  ```

- To customize embeddings:
    - Generate embeddings in the format `[idx_list, all_seq_embeddings, label_list]`. `idx_list` are the list of ontology
      term ID. `all_seq_embeddings` are the list of embedding. `label_list` are the list of names of ontology concepts.
    - Modify `build_label_tree.py` for hierarchical clustering on your embeddings.

### Pre-built Indexes

Pre-built search indexes are available in `data/{dataset}/search_index_related`.

---

## Model Training

1. Configure model settings in `scripts/configs/{dataset}/{index type}.json`.
2. Train a model:
   ```bash
   python model_train.py --index_type {index type} --dataset {dataset}
   ```
   Example:
   ```bash
   python model_train.py --index_type ssid --dataset cdr
   ```

---

## Model Evaluation

Evaluate the trained model:
```bash
python model_evaluation.py --index_type {index type} --dataset {dataset}
```
Results include precision, recall, and F1-score for passage-level queries.

---

## Model Prediction

1. Prepare query data in JSON format:
   ```json
   {
     "passage_key": {
       "query_type_0": "...",
       "query_type_1": [...],
       "query_type_2": [...]
     }
   }
   ```
   Example for HOIP:
   ```json
   {"ID72": {
    "passage": "Pyroptosis is a highly inflammatory form of lytic programmed cell death that occurs most frequently upon infection with intracellular pathogens and is likely to form part of the antimicrobial response. Pyroptosis can take place in immune cells and is also reported to occur in keratinocytes and some epithelial cells. Formation of pores causes cell membrane rupture and release of cytokines, as well as various damage-associated molecular pattern (DAMP) molecules such as HMGB-1, ATP and DNA, out of the cell. These molecules recruit more immune cells and further perpetuate the inflammatory cascade in the tissue.",
    "g_claim": [
      "Pyroptosis is a highly inflammatory form of lytic programmed cell death.",
      "Pyroptosis occurs most frequently upon infection with intracellular pathogens.",
      "Pyroptosis is likely to form part of the antimicrobial response.",
      "Pyroptosis can take place in immune cells.",
      "Pyroptosis also occurs in keratinocytes and some epithelial cells.",
      "Formation of pores causes cell membrane rupture and release of cytokines.",
      "Formation of pores causes cell membrane rupture and release of damage-associated molecular pattern (DAMP) molecules such as HMGB-1, ATP and DNA.",
      "These molecules recruit more immune cells and further perpetuate the inflammatory cascade in the tissue."
    ],
    "g_concept": [
      "programmed cell death",
      "inflammation",
      "lytic cell death",
      "release of damage-associated molecular patterns (DAMPs)",
      "cytokine production",
      "antimicrobial response"
    ]},
   ...
   }

   ```

2. Configure prediction in `scripts/configs/{dataset}/{index type}.json`:
   ```json
   {
     "prediction": true,
     "pred_path": "{your prediction file path}"
   }
   ```

3. Run the prediction script:
   ```bash
   python model_predict.py --index_type {index type} --dataset {dataset}
   ```

4. Evaluate predictions:
    - Edit `eval_multi_type_query.py` (line 176) to set query types:
      ```python
      query_type_list = ["passage", "g_claim", "g_concept"]
      ```
    - Run:
      ```bash
      python eval_multi_type_query.py --index_type {index type} --dataset {dataset} --prediction_file_path {prediction file path}
      ```

---

## Additional Information

For additional customization or evaluation options, please take a look at the provided scripts and documentation.
