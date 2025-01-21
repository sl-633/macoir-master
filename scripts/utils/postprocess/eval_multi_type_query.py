import argparse
import os
import json
import pandas as pd
import ast
from transformers import BartTokenizerFast


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


def get_eval_score(source_path, gold_path, id_name_map, output_path, query_type, k=0, thd=0, test_version="original"):
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
        "On {} set\t{}:\tTP: {:.0f}\tFP: {:.0f}\tFN: {:.0f}\tPrecision: {:.4f}\tRecall: {:.4f}\tF1: {:.4f}".format(test_version, query_type, tp,fp, fn,ent_pre * 100,ent_rec * 100,ent_f1 * 100))

    with open(output_path, "w") as f_out:
        json.dump(all_res_per_doc, f_out)
        f_out.close()


def compute_metrics(dataset, index_type, to_index_df, query_type_list, tokenizer, res_log_file_path, model_card_name, do_eval=True):
    if not os.path.exists(f"../../../data/{dataset}/cr_result"):
        os.makedirs(f"../../../data/{dataset}/cr_result")

    id_name_map = {}
    ssid_id_map = {}

    if "ssid" in index_type:
        for e_id, ent_id, label in zip(to_index_df["ssid"], to_index_df["entity_id"], to_index_df["label"]):
            e_id = ast.literal_eval(e_id)
            ssid_id_map["-".join([str(c) for c in e_id])] = ent_id
            id_name_map[ent_id] = label
    else:
        for e_id, ent_id, label in zip(to_index_df["ssid"], to_index_df["entity_id"], to_index_df["label"]):
            ssid_id_map[str(e_id)] = ent_id
            id_name_map[ent_id] = label

    ''' Get retrieval prediction results '''
    for k in [1, 5, 10]:
        out_file = f"../../../data/{dataset}/cr_result/output_{model_card_name}_top_{k}.json"
        get_pred_res(source_path=res_log_file_path, ssid_id_map=ssid_id_map, id_name_map=id_name_map,
                     output_path=out_file,
                     tokenizer=tokenizer, topk=k)
        if do_eval == "true":
            print(f"Evaluation on top - {k} generated sequences:")

            for query_type in query_type_list:
                gold_file = f"../../../data/{dataset}/ori_json/test_ori.json"
                res_out_file = f"../../../data/{dataset}/cr_result/output_{model_card_name}_{query_type}_matched.json"
                get_eval_score(source_path=out_file, gold_path=gold_file,
                               output_path=res_out_file, query_type=query_type, id_name_map=id_name_map, test_version="original")
                if dataset == "hoip":
                    gold_file = f"../../../data/{dataset}/ori_json/test_explicit_ori.json"
                    res_out_file = f"../../../data/{dataset}/cr_result/output_{model_card_name}_{query_type}_explicit_matched.json"
                    get_eval_score(source_path=out_file, gold_path=gold_file,
                                   output_path=res_out_file, query_type=query_type, id_name_map=id_name_map, test_version="explicit")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="hoip")
    parser.add_argument('--index_type', type=str, default="ssid")
    parser.add_argument('--do_eval', type=str, default="true")
    parser.add_argument('--prediction_file_path', type=str, default="log_ssid_w_name_w_synonym_pred.json")
    args, _1 = parser.parse_known_args()

    if args.index_type != "ssid" and args.dataset != "hoip":
        raise ValueError(f"Only HOIP dataset is supported with {args.index_type}.")

    index_type = args.index_type
    if index_type == "ssid":
        to_index_df = pd.read_csv(f"../../../data/{args.dataset}/search_index_related/to_ssid_k10_c10.csv", sep='\t')
    elif index_type == "randomid":
        to_index_df = pd.read_csv(f"../../../data/{args.dataset}/search_index_related/to_random_id.csv", sep='\t')
    elif index_type == "ssid_w_hypernym":
        to_index_df = pd.read_csv(f"../../../data/{args.dataset}/search_index_related/to_ssid_w_hypernym_k10_c10.csv",
                                  sep='\t')
    elif index_type == "ontoid":
        to_index_df = pd.read_csv(f"../../../data/{args.dataset}/search_index_related/to_onto_id_.csv", sep='\t')
    else:
        raise ValueError("Not preferred index type.")
    input_file_path = f"../../logs/{args.dataset}/recognition/{args.prediction_file_path}"
    tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large")
    query_type_list = ["passage", "g_claim", "g_concept"]

    compute_metrics(dataset=args.dataset, index_type=index_type, to_index_df=to_index_df, tokenizer=tokenizer,
                    res_log_file_path=input_file_path,
                    query_type_list=query_type_list, do_eval=args.do_eval,
                    model_card_name="_".join(args.prediction_file_path[:-5].split("_")[1:-1]))
    print("Finished.")