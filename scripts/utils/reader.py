import os
import json

from datasets import Dataset, DatasetDict


def load_passage_pairs(source_path: str, tokenizer, max_seq_length=1024, cate="test"):
    all_insts = []
    final_data = {}

    max_src_len, max_tgt_len = -1, -1
    with open(os.path.join(source_path), "r") as f_in:
        ori_insts = json.load(f_in)
        f_in.close()

    doc_counter = 0

    for inst in ori_insts:

        input_text = inst["description"]
        search_id = []
        for e in inst["entity_list"]:
            if type(e) == list:
                cur_search_id = "-".join([str(idx) for idx in e])
            else:
                cur_search_id = str(e)
            search_id.append(cur_search_id)
        search_id = sorted(search_id)
        search_id = ";".join(search_id)
        search_id += ";"

        src_ids = tokenizer(input_text.strip(), return_tensors='pt')['input_ids'][0].tolist()
        if len(src_ids) > max_seq_length:
            src_ids = src_ids[:max_seq_length - 1] + [tokenizer.eos_token_id]
        tgt_ids = tokenizer(search_id.strip(), return_tensors='pt')['input_ids'][0].tolist()

        if len(tgt_ids) > max_seq_length:
            tgt_ids = tgt_ids[:max_seq_length - 1] + [tokenizer.eos_token_id]

        new_inst = {
            'doc_key': inst['doc_key'],
            'text': inst['description'],
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
        }

        all_insts.append(new_inst)
        doc_counter += 1
        del new_inst

    # all_insts = all_insts[:16]
    final_data[cate] = Dataset.from_list(all_insts)
    src_lens = [len(c["src_ids"]) for c in all_insts]
    max_src_len = max(max_src_len, max(src_lens))
    tgt_lens = [len(c["tgt_ids"]) for c in all_insts]
    max_tgt_len = max(max_tgt_len, max(tgt_lens))

    print(
        f"Load in {len(all_insts)} passage-index instances for {cate} set.  (max_src_len={max_src_len}, max_tgt_len={max_tgt_len})")
    del all_insts

    return DatasetDict(final_data), max_src_len, max_tgt_len


def load_span_pairs(source_path: str, tokenizer, max_seq_length=1024, cate="train"):
    all_insts = []

    max_src_len, max_tgt_len = -1, -1
    with open(os.path.join(source_path), "r") as f_in:
        ori_insts = json.load(f_in)
        f_in.close()

    doc_counter = 0

    for inst in ori_insts:

        input_text = inst["label"]
        search_id = str(inst["si"])

        src_ids = tokenizer(input_text.strip(), return_tensors='pt')['input_ids'][0].tolist()
        if len(src_ids) > max_seq_length:
            src_ids = src_ids[:max_seq_length - 1] + [tokenizer.eos_token_id]
        tgt_ids = tokenizer(search_id.strip(), return_tensors='pt')['input_ids'][0].tolist()
        if len(tgt_ids) > max_seq_length:
            tgt_ids = tgt_ids[:max_seq_length - 1] + [tokenizer.eos_token_id]

        new_inst = {
            'doc_key': inst['concept_id'],
            'text': inst['label'],
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
        }

        all_insts.append(new_inst)
        doc_counter += 1
        del new_inst

    src_lens = [len(c["src_ids"]) for c in all_insts]
    max_src_len = max(max_src_len, max(src_lens))
    tgt_lens = [len(c["tgt_ids"]) for c in all_insts]
    max_tgt_len = max(max_tgt_len, max(tgt_lens))

    print(
        f"Load in {len(all_insts)} span-index instances for {cate} set.  (max_src_len={max_src_len}, max_tgt_len={max_tgt_len})")

    return all_insts, max_src_len, max_tgt_len


def load_text(dataset: str, source_path: str, tokenizer, max_seq_length=1024):
    if dataset == "cdr":
        all_insts, max_src_len, max_tgt_len = load_cdr_text(source_path, tokenizer, max_seq_length)
    elif dataset == "hpo":
        all_insts, max_src_len, max_tgt_len = load_hpo_text(source_path, tokenizer, max_seq_length)
    elif dataset == "hoip":
        all_insts, max_src_len, max_tgt_len = load_hoip_text(source_path, tokenizer, max_seq_length)
    else:
        raise ValueError("The name of dataset should be 'cdr', 'hpo' or 'hoip'.")
    return all_insts, max_src_len, max_tgt_len


def load_hoip_text(source_path: str, tokenizer, max_seq_length=1024):
    all_insts = []

    max_src_len, max_tgt_len = -1, -1
    with open(os.path.join(source_path), "r") as f_in:
        ori_insts = json.load(f_in)
        f_in.close()

    doc_counter = 0

    for d_id, inst in ori_insts.items():
        if "passage" in inst.keys():

            input_text = inst["passage"]

            src_ids = tokenizer(input_text.strip(), return_tensors='pt')['input_ids'][0].tolist()
            if len(src_ids) > max_seq_length:
                src_ids = src_ids[:max_seq_length - 1] + [tokenizer.eos_token_id]

            new_inst = {
                'doc_key': d_id,
                'doc_type': "passage",
                'text': input_text,
                'src_ids': src_ids,
            }

            all_insts.append(new_inst)
            doc_counter += 1
            del new_inst
        if "g_claim" in inst.keys():
            for input_text in inst["g_claim"]:

                src_ids = tokenizer(input_text.strip(), return_tensors='pt')['input_ids'][0].tolist()
                if len(src_ids) > max_seq_length:
                    src_ids = src_ids[:max_seq_length - 1] + [tokenizer.eos_token_id]

                new_inst = {
                    'doc_key': d_id,
                    'doc_type': "g_claim",
                    'text': input_text,
                    'src_ids': src_ids,
                }

                all_insts.append(new_inst)
                doc_counter += 1
                del new_inst

        if "g_concept" in inst.keys():
            for input_text in inst["g_concept"]:

                src_ids = tokenizer(input_text.strip(), return_tensors='pt')['input_ids'][0].tolist()
                if len(src_ids) > max_seq_length:
                    src_ids = src_ids[:max_seq_length - 1] + [tokenizer.eos_token_id]

                new_inst = {
                    'doc_key': d_id,
                    'doc_type': "g_concept",
                    'text': input_text,
                    'src_ids': src_ids,
                }

                all_insts.append(new_inst)
                doc_counter += 1
                del new_inst
    src_lens = [len(c["src_ids"]) for c in all_insts]
    max_src_len = max(max_src_len, max(src_lens))

    print(
        f"Load in {len(all_insts)} instances for prediction.  (max_src_len={max_src_len})")

    return all_insts, max_src_len, max_tgt_len


def load_cdr_text(source_path: str, tokenizer, max_seq_length=1024):
    all_insts = []

    max_src_len, max_tgt_len = -1, -1
    with open(os.path.join(source_path), "r") as f_in:
        ori_insts = json.load(f_in)
        f_in.close()

    doc_counter = 0

    for d_id, inst in ori_insts.items():
        if "passage" in inst.keys():

            input_text = inst["passage"]

            src_ids = tokenizer(input_text.strip(), return_tensors='pt')['input_ids'][0].tolist()
            if len(src_ids) > max_seq_length:
                src_ids = src_ids[:max_seq_length - 1] + [tokenizer.eos_token_id]

            new_inst = {
                'doc_key': d_id,
                'doc_type': "passage",
                'text': input_text,
                'src_ids': src_ids,
            }

            all_insts.append(new_inst)
            doc_counter += 1
            del new_inst
        if "mention" in inst.keys():
            for input_text in inst["mention"]:

                src_ids = tokenizer(input_text.strip(), return_tensors='pt')['input_ids'][0].tolist()
                if len(src_ids) > max_seq_length:
                    src_ids = src_ids[:max_seq_length - 1] + [tokenizer.eos_token_id]

                new_inst = {
                    'doc_key': d_id,
                    'doc_type': "mention",
                    'text': input_text,
                    'src_ids': src_ids,
                }

                all_insts.append(new_inst)
                doc_counter += 1
                del new_inst

        if "g_concept" in inst.keys():
            for input_text in inst["g_concept"]:

                src_ids = tokenizer(input_text.strip(), return_tensors='pt')['input_ids'][0].tolist()
                if len(src_ids) > max_seq_length:
                    src_ids = src_ids[:max_seq_length - 1] + [tokenizer.eos_token_id]

                new_inst = {
                    'doc_key': d_id,
                    'doc_type': "g_concept",
                    'text': input_text,
                    'src_ids': src_ids,
                }

                all_insts.append(new_inst)
                doc_counter += 1
                del new_inst

    src_lens = [len(c["src_ids"]) for c in all_insts]
    max_src_len = max(max_src_len, max(src_lens))

    print(
        f"Load in {len(all_insts)} instances for prediction.  (max_src_len={max_src_len})")

    return all_insts, max_src_len, max_tgt_len


def load_hpo_text(source_path: str, tokenizer, max_seq_length=1024):
    all_insts = []

    max_src_len, max_tgt_len = -1, -1
    with open(os.path.join(source_path), "r") as f_in:
        ori_insts = json.load(f_in)
        f_in.close()

    doc_counter = 0

    for d_id, inst in ori_insts.items():
        if "passage" in inst.keys():

            input_text = inst["passage"]

            src_ids = tokenizer(input_text.strip(), return_tensors='pt')['input_ids'][0].tolist()
            if len(src_ids) > max_seq_length:
                src_ids = src_ids[:max_seq_length - 1] + [tokenizer.eos_token_id]

            new_inst = {
                'doc_key': d_id,
                'doc_type': "passage",
                'text': input_text,
                'src_ids': src_ids,
            }

            all_insts.append(new_inst)
            doc_counter += 1
            del new_inst
        if "mention" in inst.keys():
            for input_text in inst["mention"]:

                src_ids = tokenizer(input_text.strip(), return_tensors='pt')['input_ids'][0].tolist()
                if len(src_ids) > max_seq_length:
                    src_ids = src_ids[:max_seq_length - 1] + [tokenizer.eos_token_id]

                new_inst = {
                    'doc_key': d_id,
                    'doc_type': "mention",
                    'text': input_text,
                    'src_ids': src_ids,
                }

                all_insts.append(new_inst)
                doc_counter += 1
                del new_inst

        if "g_concept" in inst.keys():
            for input_text in inst["g_concept"]:

                src_ids = tokenizer(input_text.strip(), return_tensors='pt')['input_ids'][0].tolist()
                if len(src_ids) > max_seq_length:
                    src_ids = src_ids[:max_seq_length - 1] + [tokenizer.eos_token_id]

                new_inst = {
                    'doc_key': d_id,
                    'doc_type': "g_concept",
                    'text': input_text,
                    'src_ids': src_ids,
                }

                all_insts.append(new_inst)
                doc_counter += 1
                del new_inst

    src_lens = [len(c["src_ids"]) for c in all_insts]
    max_src_len = max(max_src_len, max(src_lens))

    print(
        f"Load in {len(all_insts)} instances for prediction.  (max_src_len={max_src_len})")

    return all_insts, max_src_len, max_tgt_len
