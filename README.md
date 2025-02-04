# MA-COIR

## Overview

This repository contains tools and workflows for data preprocessing, label tree building, model training, evaluation,
and prediction for Mention-Agnostic Concept Recognition through an Indexing-Recognition Framework (MA-COIR).
![github_overview](https://github.com/user-attachments/assets/6b045a91-8987-4d8c-b7ac-680ce8544847)

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
    - Download the dataset from [HPO GSC+](https://github.com/lasigeBioTM/IHP/blob/master/GSC%2B.rar).
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
![github_indexing](https://github.com/user-attachments/assets/aa39c46f-cc60-48cb-9a40-6840dafcb3e7)

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

## Generate multiple types of queries

The large language model we used is `meta-llama/Meta-Llama-3-8B-Instruct`.
The configuration of generation is
```python
def generate_per_inst(model, terminators, input_ids):
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=model.config.pad_token_id
    )
    response = outputs[0][input_ids.shape[-1]:]

    return tokenizer.decode(response, skip_special_tokens=True)
```

Here we provide the prompt templates used in the paper for claim and concept name generation given a passage.

### CDR concept
```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful, respectful and honest assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Please list all concepts referring to a chemical concept or a disease concept in following input (make sure not to add any information), and return the output as a jsonl, where each line is {"Chemical":[CHEMICAL]} or {"Disease":[DISEASE]}. If there is no chemical or disease concept in the input, it is fine to only return {"Chemical":None} or {"Disease":None}. Directly return the jsonl with no explanation or other formatting.
The input is: "{passage for the 1st demostration}" <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{answer for the 1st demostration} <|eot_id|>
…
<|start_header_id|>user<|end_header_id|>
The input is: "{passage for the nth demostration}"<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{answer for the nth demostration} <|eot_id|>
<|start_header_id|>user<|end_header_id|>
Please list all concepts referring to a chemical concept or a disease concept in following input (make sure not to add any information), and return the output as a jsonl, where each line is {"Chemical":[CHEMICAL]} or {"Disease":[DISEASE]}. If there is no chemical or disease concept in the input, it is fine to only return {"Chemical":None} or {"Disease":None}. Directly return the jsonl with no explanation or other formatting.
The input is: "{passage}" <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

### HPO concept
```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful, respectful and honest assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Please list all concepts referring to a medically relevant human phenotype concept in following input (make sure not to add any information), and return the output as a jsonl, where each line is {"phenotype":[PHENOTYPE]}. If there is no human phenotype concept in the input, it is fine to only return {"phenotype":None}. Directly return the jsonl with no explanation or other formatting. 
The input is: "{passage for the 1st demostration}" <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{answer for the 1st demostration} <|eot_id|>
…
<|start_header_id|>user<|end_header_id|>
The input is: "{passage for the nth demostration}"<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{answer for the nth demostration} <|eot_id|>
<|start_header_id|>user<|end_header_id|>
Please list all concepts referring to a medically relevant human phenotype concept in following input (make sure not to add any information), and return the output as a jsonl, where each line is {"phenotype":[PHENOTYPE]}. If there is no human phenotype concept in the input, it is fine to only return {"phenotype":None}. Directly return the jsonl with no explanation or other formatting. 
The input is: "{passage}" <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

### HOIP concept
```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful, respectful and honest assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Please list all biological processes involved in the phenomenon described in the following input (make sure not to add any information), and return the output as a jsonl, where each line is {"process":[PROCESS]}. If there is no process in the input, it is fine to only return {"process":None}. Directly return the jsonl with no explanation or other formatting.
The input is: "{passage for the 1st demostration}" <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{answer for the 1st demostration} <|eot_id|>
…
<|start_header_id|>user<|end_header_id|>
The input is: "{passage for the nth demostration}"<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{answer for the  nth demostration} <|eot_id|>
<|start_header_id|>user<|end_header_id|>
Please list all biological processes involved in the phenomenon described in the following input (make sure not to add any information), and return the output as a jsonl, where each line is {"process":[PROCESS]}. If there is no process in the input, it is fine to only return {"process":None}. Directly return the jsonl with no explanation or other formatting.
The input is: "{passage}" <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

### HOIP claim
```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful, respectful and honest assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Please breakdown the following input into a set of small, independent claims (make sure not to add any information), and return the output as a jsonl, where each line is {"claim":[CLAIM], "score":[CONF]}. The confidence score [CONF] should represent your confidence in the claim, where a 1 is obvious facts and results like ‘The earth is round’ and ‘1+1=2’. A 0 is for claims that are very obscure or difficult for anyone to know, like the birthdays of non-notable people. If the input is short, it is fine to only return 1 claim. Directly return the jsonl with no explanation or other formatting. The input is: "{passage}" <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```
It is noted that LLM-generated outputs do not always follow the format limitation. Some manual efforts are needed to post-process the generated outputs.

We provide scripts for LLM-based generation tasks.

- `generate_claim_query.py`: Generates claims based on the dataset and input file.
- `generate_concept_query.py`: Generates concepts based on the dataset and input file.

To run these scripts, you need to:

1. Place your Hugging Face Hub API key in **line 51** of the respective script.
2. Set the `dataset` and `passage_file` parameters when executing the script.

### Generating Claims
Run the `generate_claim_query.py` script as follows:
```bash
python generate_claim_query.py --dataset {dataset} --passage_file {passage_file}
```

#### Example
To generate claims for the HOIP test passages:
```bash
python generate_claim_query.py --dataset hoip --passage_file test_ori.json
```

The model outputs will be saved in the directory: ```../../../data/{dataset}/generated_files```

### Generating Concepts
Run the `generate_concept_query.py` script as follows:
```bash
python generate_concept_query.py --dataset {dataset} --passage_file {passage_file}
```

In our experiments, we used **10-shot In-Context Learning (ICL)** for concept name generation. Demonstrations were randomly selected from the training set. Note that we do not provide a specific demonstration selection code in `generate_claim_query.py`. Instead, we recommend designing your own demonstration selection strategies, using our zero-shot generation scripts as a starting point.

---

## Data augmentation
![github_claim_concept](https://github.com/user-attachments/assets/0cc16a43-287b-42b6-b44a-d0c03427c12c)

We provide the scripts for text excerpt mining in `scripts/utils/preprocess/get_excerpts.py`.

Set the `dataset`, `input_type`, `input_file_path`, and `output_file_path` parameters when executing the script.

`input_type` can be `passage` or `claim`. When `input_type=passage`, we first segmented a passage into sentences, then
extracted text excerpts. When `input_type=claim`, we load `g_claim` list for each passage from the input file, then get
the text excerpts from each claim. All excerpts from claims of one passage are merged into a list and the list is saved
as `excerpt` in the output file.

For example, to get excerpts for the HOIP test passages:

```bash
python generate_claim_query.py --dataset hoip --input_type passage --input_file_path test_ori.json --output_file_path test_excerpt.json
```

The scripts used for the labelling function when constructing the claim-concept data are not included in this repository.
This is because this step involves a simple cosine-similarity calculation between excerpts and gold concepts, which can
be implemented using any method or tool of your choice.

The claim-concept data we constructed is available in the following file:
`data/hoip/ori_json/train_claim_concept_pairs.json`.


---

## Additional Information

For additional customization or evaluation options, please take a look at the provided scripts and documentation.
