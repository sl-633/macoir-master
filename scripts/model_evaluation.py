import argparse
import os

import torch
import json
import pandas as pd
import ast

from transformers import BartTokenizerFast, BartConfig, BartModel
from networks.macoir import MACOIR
from torch.utils.data import DataLoader
from utils.reader import load_passage_pairs


# Save and Load Functions

def save_checkpoint(save_path, model, valid_loss):
    if save_path is None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, device):
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def get_pred_res(source_path: str, ssid_id_map, id_name_map, output_path: str, tokenizer, topk=10):
    with open(source_path, "r") as f_in:
        all_d = json.load(f_in)
        f_in.close()

    new_d = []
    len_pred = 0
    for d in all_d:

        prediction_list = tokenizer.batch_decode(d[3], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        prediction = []
        for p in prediction_list[:topk]:
            prediction.extend(p.strip().split(";"))
        prediction = list(set(prediction))
        prediction = [p for p in prediction if p != ""]
        len_pred += len(prediction)
        new_p = []

        for p in prediction:
            if len(p) < 2:
                continue
            new_p.append({
                "gen_id": p,
                "ent_id": "None" if p not in ssid_id_map.keys() else ssid_id_map[p],
                "name": "None" if (p not in ssid_id_map.keys() or ssid_id_map[p] not in id_name_map.keys()) else
                id_name_map[ssid_id_map[p]],
            })
        new_d.append({
            "doc_id": d[0],
            "doc_type": d[2],
            "pred": new_p
        })
    with open(output_path, "w") as f_out:
        json.dump(new_d, f_out)
        f_out.close()


def get_eval_score(source_path, gold_path, id_name_map, output_path, query_type, k=0, thd=0):
    with open(source_path, "r") as f_in:
        all_d = json.load(f_in)
        f_in.close()
    pred_per_doc = {}
    all_res_per_doc = {}
    for d in all_d:
        pred_per_doc[d["doc_id"]] = []
    for d in all_d:
        if d["doc_type"] != query_type:
            continue
        if k != 0:
            d["pred"] = d["pred"][:k]
        for p in d["pred"]:
            if p["ent_id"] != "None":
                if thd != 0 and p["score"] < thd:
                    continue
                pred_per_doc[d["doc_id"]].append(p["ent_id"])

    with open(gold_path, "r") as f_in:
        all_d = json.load(f_in)
        f_in.close()

    gold_per_doc = {}
    for d_id, d in all_d.items():
        gold_per_doc[d_id] = []
        for e in d["entities"]:
            if e["id"] not in id_name_map.keys():
                continue
            gold_per_doc[d_id].append(e["id"])
        gold_per_doc[d_id] = list(set(gold_per_doc[d_id]))

    tp, fp, fn = 0.0, 0.0, 0.0
    for d_id, g_ents in gold_per_doc.items():

        if d_id not in pred_per_doc.keys():
            pred_per_doc[d_id] = []
        else:
            pred_per_doc[d_id] = list(set(pred_per_doc[d_id]))
        tp_list = [e for e in pred_per_doc[d_id] if e in g_ents]
        fp_list = [e for e in pred_per_doc[d_id] if e not in g_ents]
        fn_list = [e for e in g_ents if e not in pred_per_doc[d_id]]
        # all_res_per_doc[d_id]["TP_Entity"] = [{"id": ssid_id_map[e], "name": ssid_label_map[e]} for e in tp_list]
        # all_res_per_doc[d_id]["FN_Entity"] = [{"id": ssid_id_map[e], "name": ssid_label_map[e]} for e in fn_list]
        # all_res_per_doc[d_id]["Matched_Entity"] = [{"id": ssid_id_map[e], "name": ssid_label_map[e]} for e in
        #                                             pred_per_doc[d_id]]
        all_res_per_doc[d_id] = {
            "passage": all_d[d_id]["passage"],
            "gold_entities": [{"id": e, "name": id_name_map[e]} for e in g_ents],
            "pred_entities": [{"id": e, "name": id_name_map[e]} for e in pred_per_doc[d_id]]
        }
        tp += len(tp_list)
        fp += len(fp_list)
        fn += len(fn_list)
        all_res_per_doc[d_id]["TP"] = len(tp_list)

    ent_pre = tp / (tp + fp) if tp + fp > 0.0 else 0.0
    ent_rec = tp / (tp + fn) if tp + fn > 0.0 else 0.0
    ent_f1 = 2 * ent_pre * ent_rec / (ent_pre + ent_rec) if ent_pre + ent_rec > 0.0 else 0.0
    print(
        "{}:\tTP: {:.0f}\tFP: {:.0f}\tFN: {:.0f}\tPrecision: {:.4f}\tRecall: {:.4f}\tF1: {:.4f}".format(query_type, tp,
                                                                                                        fp, fn,
                                                                                                        ent_pre * 100,
                                                                                                        ent_rec * 100,
                                                                                                        ent_f1 * 100))

    with open(output_path, "w") as f_out:
        json.dump(all_res_per_doc, f_out)
        f_out.close()


def compute_metrics(dataset, index_type, tokenizer, res_log_file_path, model_card_name):
    if index_type == "ssid":
        to_ssid_df = pd.read_csv(f"../data/{dataset}/search_index_related/to_ssid_k10_c10.csv", sep='\t')
    elif index_type == "randomid":
        to_ssid_df = pd.read_csv(f"../data/{dataset}/search_index_related/to_random_id.csv", sep='\t')
    elif index_type == "ssid_w_hypernym":
        to_ssid_df = pd.read_csv(f"../data/{dataset}/search_index_related/to_ssid_w_hypernym_k10_c10.csv", sep='\t')
    elif index_type == "ontoid":
        to_ssid_df = pd.read_csv(f"../data/{dataset}/search_index_related/to_onto_id.csv", sep='\t')
    else:
        raise ValueError("Not preferred index type.")

    if not os.path.exists(f"../data/{dataset}/cr_result"):
        os.makedirs(f"../data/{dataset}/cr_result")

    id_name_map = {}
    ssid_id_map = {}

    if "ssid" in index_type:
        for e_id, ent_id, label in zip(to_ssid_df["ssid"], to_ssid_df["entity_id"], to_ssid_df["label"]):
            e_id = ast.literal_eval(e_id)
            ssid_id_map["-".join([str(c) for c in e_id])] = ent_id
            id_name_map[ent_id] = label
    else:
        for e_id, ent_id, label in zip(to_ssid_df["ssid"], to_ssid_df["entity_id"], to_ssid_df["label"]):
            ssid_id_map[str(e_id)] = ent_id
            id_name_map[ent_id] = label

    ''' Get retrieval prediction results '''
    for k in [1, 5, 10]:
        out_file = f"../data/{dataset}/cr_result/output_{model_card_name}.json"
        get_pred_res(source_path=res_log_file_path, ssid_id_map=ssid_id_map, id_name_map=id_name_map,
                     output_path=out_file,
                     tokenizer=tokenizer, topk=k)
        print(f"Evaluation on top - {k} generated sequences:")

        query_type_list = ["passage"]
        for query_type in query_type_list:
            gold_file = f"../data/{dataset}/ori_json/test_ori.json"
            res_out_file = f"../data/{dataset}/cr_result/output_{model_card_name}_{query_type}_matched.json"
            get_eval_score(source_path=out_file, gold_path=gold_file,
                           output_path=res_out_file, query_type=query_type, id_name_map=id_name_map)
            if dataset == "hoip":
                gold_file = f"../data/{dataset}/ori_json/test_explicit_ori.json"
                res_out_file = f"../data/{dataset}/cr_result/output_{model_card_name}_{query_type}_explicit_matched.json"
                get_eval_score(source_path=out_file, gold_path=gold_file,
                               output_path=res_out_file, query_type=query_type, id_name_map=id_name_map)


def do_eval(model, dev_loader, dataset, index_type, device, log_file_path, model_card_name, num_return_sequences=10):
    truths = []
    predictions = []
    ent_ids = []
    ent_texts = []
    id_max_length = 300

    def _pad_tensors_to_max_len(tensor, max_length):
        if tokenizer is not None and hasattr(tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
        else:
            if model.config.pad_token_id is not None:
                pad_token_id = model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    with torch.no_grad():
        for v_src_ids, v_src_mask, v_tgt_ids, v_labels, v_ent_ids, v_ent_texts in dev_loader:

            inputs = {
                'input_ids': v_src_ids.to(device),
                'attention_mask': v_src_mask.to(device),
            }

            batch_beams = model.generate(
                inputs['input_ids'].to(device),
                max_length=id_max_length,
                num_beams=num_return_sequences,
                prefix_allowed_tokens_fn=restrict_decode_vocab,
                num_return_sequences=num_return_sequences,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                early_stopping=True)

            if batch_beams.shape[-1] < id_max_length:
                batch_beams = _pad_tensors_to_max_len(batch_beams, id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], num_return_sequences, -1)
            output_tokens = batch_beams[:, :, 1:].tolist()
            predictions.extend(output_tokens)

            truths.extend(v_labels[:, 1:].cpu().tolist())

            ent_ids.extend(v_ent_ids)
            ent_texts.extend(v_ent_texts)

    res = []
    for p, l, ent_id, ent_text in zip(predictions, truths, ent_ids, ent_texts):
        res.append([ent_id, ent_text, l, p, 1 if l == p else 0])
    res_log_file_path = log_file_path + f"/log_{model_card_name}_eval.json"
    with open(res_log_file_path, "w") as f_out:
        json.dump(res, f_out)
        f_out.close()

    compute_metrics(dataset=dataset, index_type=index_type, tokenizer=tokenizer, res_log_file_path=res_log_file_path,
                    model_card_name=model_card_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_type', type=str, default="ssid")
    parser.add_argument('--dataset', type=str, default="cdr")

    args, _1 = parser.parse_known_args()

    args.config = f"configs/{args.dataset}/{args.index_type}.json"

    if args.config is not None:
        print("Loading the configuration from " + args.config)
        with open(args.config, "r") as f_in:
            config_dict = json.load(f_in)
            args.__dict__.update(config_dict)
            f_in.close()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    args.n_gpu = torch.cuda.device_count()

    args.log_dir = f"logs/{args.dataset}/recognition"
    args.model_dir = f"models/{args.dataset}/recognition"

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    tokenizer = BartTokenizerFast.from_pretrained(args.transformer_name)

    transformers_configs = BartConfig(args.transformer_name)
    transformer_model = BartModel.from_pretrained(args.transformer_name)

    model = MACOIR(config=transformers_configs, model=transformer_model, tokenizer=tokenizer)

    model.config.return_dict = True

    cur_best_f1 = 0.0

    # entid generation constrain, we only generate integer entids.
    SPIECE_HYPEN = "-"
    SPIECE_SEP = ";"
    INT_TOKEN_IDS = []
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_HYPEN:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_HYPEN:
            INT_TOKEN_IDS.append(id)
        elif token == SPIECE_SEP:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit() and len(token) <= 2:
            INT_TOKEN_IDS.append(id)
        elif token in ['HO', 'IP', '_', 'GO']:
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)
    print(f"The decoding vocabulary size is {len(INT_TOKEN_IDS)}")

    INT_TOKEN_IDS = sorted(INT_TOKEN_IDS)


    def restrict_decode_vocab():
        return INT_TOKEN_IDS


    def collate_fn(batch):
        src_ids = [f["src_ids"] + [tokenizer.pad_token_id] * (max_src_len - len(f["src_ids"])) for f in batch]
        src_mask = [[1.0] * len(f["src_ids"]) + [0.0] * (max_src_len - len(f["src_ids"])) for f in batch]

        ent_ids = [f["doc_key"] for f in batch]
        ent_texts = [f["text"] for f in batch]
        labels = [f["tgt_ids"] + [tokenizer.pad_token_id] * (max_tgt_len - len(f["tgt_ids"])) for f in batch]
        tgt_ids = [f["tgt_ids"] + [tokenizer.pad_token_id] * (max_tgt_len - len(f["tgt_ids"])) for f in batch]

        src_ids = torch.tensor(src_ids, dtype=torch.long)
        tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        src_mask = torch.tensor(src_mask, dtype=torch.float)
        return src_ids, src_mask, tgt_ids, labels, ent_ids, ent_texts


    args.model_card_name = args.index_type
    if args.use_name:
        args.model_card_name = args.model_card_name + "_w_name"

    if args.use_synonym:
        args.model_card_name = args.model_card_name + "_w_synonym"

    if args.use_claim:
        args.model_card_name = args.model_card_name + "_w_claim"

    print(f"Loaded model_{args.model_card_name}.pt for subsequent model evaluation...")

    if args.evaluation:
        load_checkpoint(f"models/{args.dataset}/recognition/model_{args.model_card_name}.pt", model, device=args.device)
        model.eval()
        # Set model
        model.to(args.device)
        print("Now the evaluation is running...")
        all_data, max_src_len, max_tgt_len = load_passage_pairs(args.test_passage_path, tokenizer, cate="test",
                                                                max_seq_length=1024)

        dev_loader = DataLoader(all_data["test"], batch_size=args.per_device_train_batch_size,
                                shuffle=False,
                                collate_fn=collate_fn, drop_last=False)

        do_eval(model=model, dev_loader=dev_loader, dataset=args.dataset, index_type=args.index_type,
                device=args.device, log_file_path=args.log_dir, model_card_name=args.model_card_name,
                num_return_sequences=args.num_return_sequences)
    print('Finished.')
