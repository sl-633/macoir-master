import argparse
import ast
import json
import os
import random
import re

import numpy as np
import pandas as pd


def get_hoip_passage_id_pairs(index_type: str, id_to_index_file: str):
    with open(id_to_index_file, "r") as f_in:
        id_to_index_map = json.load(f_in)
        f_in.close()

    cate_list = ["train", "dev", "test"]
    for cate in cate_list:
        max_ent_num = 0
        with open(f"../../../data/hoip/ori_json/{cate}_ori.json", "r") as f_in:
            all_d = json.load(f_in)
            f_in.close()
        new_d_list = []
        count_ent = 0
        for d_id, d in all_d.items():
            new_d = {
                "doc_key": d_id,
                "description": d["passage"]
            }
            ent_list = []
            showed_ids = []
            # print(len(new_d_list))
            for e in d["entities"]:
                if e["id"] not in showed_ids:
                    if e["id"] not in id_to_index_map.keys():
                        continue
                    else:
                        ent_list.append(id_to_index_map[e["id"]])
                        showed_ids.append(e["id"])
            count_ent += len(ent_list)
            max_ent_num = max(max_ent_num, len(ent_list))
            new_d["entity_list"] = ent_list
            new_d_list.append(new_d)
        with open(f"../../../data/hoip/cr_input_json/{index_type}_{cate}.json", "w") as f_out:
            json.dump(new_d_list, f_out)
            f_out.close()
        print(f"{len(new_d_list)} abstracts ({count_ent} entities) are processed for {cate} set.")
        print(f"One instance contains most at {max_ent_num} entities for {cate} set.")


def get_hoip_span_id_pairs(index_type: str, id_to_index_file: str):
    with open(f"../../../data/hoip/ori_json/train_ori.json", "r") as f_in:
        all_d = json.load(f_in)
        f_in.close()

    with open(id_to_index_file, "r") as f_in:
        id_to_index_map = json.load(f_in)
        f_in.close()

    mention_id_dict = {}
    showed_id = []
    for d_id, d in all_d.items():
        for e in d["entities"]:
            # print(e)

            if e["name"] not in mention_id_dict.keys():
                mention_id_dict[e["name"]] = []

            if e["id"] not in id_to_index_map.keys() and "|" in e["id"]:
                for ee_id in e["id"].split("|"):
                    if ee_id in id_to_index_map.keys() and ee_id not in mention_id_dict[e["name"]]:
                        mention_id_dict[e["name"]].append(ee_id)
                        showed_id.append(ee_id)
            elif e["id"] in id_to_index_map.keys() and e["id"] not in mention_id_dict[e["name"]]:
                showed_id.append(e["id"])
                mention_id_dict[e["name"]].append(e["id"])

    with open(f"../../../data/hoip/ori_json/hoip_ontology.json", "r") as f_in:
        id_to_ent_map = json.load(f_in)
        f_in.close()

    name_id_dict = {}
    synonym_id_dict = {}
    count_c_name = 0
    for e_id, e in id_to_ent_map.items():
        if e_id not in showed_id:
            continue
        count_c_name += 1
        if e[0] not in name_id_dict.keys():
            name_id_dict[e[0]] = [e_id]
        else:
            if e_id not in name_id_dict[e[0]]:
                name_id_dict[e[0]].append(e_id)
        for s in e[-1]:
            if s not in synonym_id_dict.keys():
                synonym_id_dict[s] = [e_id]
            else:
                if e_id not in synonym_id_dict[s]:
                    synonym_id_dict[s].append(e_id)

    # Name - id pairs
    new_d_list = []
    for name, id_list in name_id_dict.items():

        ssid = ""
        for e_id in id_list:
            if index_type == "ssid" or "ssid_w_hypernym":
                ssid += "-".join([str(x) for x in id_to_index_map[e_id]])
            else:
                ssid = str(id_to_index_map[e_id])
            ssid += ";"
        new_d = {
            "ent_id": "|".join(id_list),
            "label": name,
            "ssid": ssid
        }
        # if len(id_list) > 1:
        #     print(new_d)
        new_d_list.append(new_d)

    with open(f"../../../data/hoip/cr_input_json/{index_type}_train_name_to_id.json", "w") as f_out:
        json.dump(new_d_list, f_out)
        f_out.close()
    print(
        f"{len(new_d_list)} name - ssid pairs ({count_c_name} entities) are processed for cr task model training.")

    # Synonym - id pairs
    new_d_list = []
    for name, id_list in synonym_id_dict.items():

        ssid = ""
        for e_id in id_list:
            if index_type == "ssid" or "ssid_w_hypernym":
                ssid += "-".join([str(x) for x in id_to_index_map[e_id]])
            else:
                ssid = str(id_to_index_map[e_id])
            ssid += ";"
        new_d = {
            "ent_id": "|".join(id_list),
            "label": name,
            "ssid": ssid
        }
        # if len(id_list) > 1:
        #     print(new_d)
        new_d_list.append(new_d)

    with open(f"../../../data/hoip/cr_input_json/{index_type}_train_synonym_to_id.json", "w") as f_out:
        json.dump(new_d_list, f_out)
        f_out.close()
    print(
        f"{len(new_d_list)} synonym - ssid pairs ({count_c_name} entities) are processed for cr task model training.")


def get_hpo_passage_id_pairs(index_type: str, id_to_index_file: str):
    with open(id_to_index_file, "r") as f_in:
        id_to_index_map = json.load(f_in)
        f_in.close()

    cate_list = ["train", "dev", "test"]
    for cate in cate_list:
        max_ent_num = 0
        with open(f"../../../data/hpo/ori_json/{cate}_ori.json", "r") as f_in:
            all_d = json.load(f_in)
            f_in.close()
        new_d_list = []
        count_ent = 0
        for d_id, d in all_d.items():
            new_d = {
                "doc_key": d_id,
                "description": d["abstract"]
            }
            ent_list = []
            showed_ids = []
            # print(len(new_d_list))
            for e in d["entities"]:
                if e["id"] not in showed_ids:
                    if e["id"] not in id_to_index_map.keys() and "|" in e["id"]:
                        for ee_id in e["id"].split("|"):
                            if ee_id not in showed_ids and ee_id != "-1":
                                ent_list.append(id_to_index_map[ee_id])
                                showed_ids.append(ee_id)
                    else:

                        # if e["id"] in new_map.keys():
                        #     e["id"] = new_map[e["id"]]
                        if "http://purl.obolibrary.org/obo/" + e["id"] not in showed_ids:
                            ent_list.append(id_to_index_map["http://purl.obolibrary.org/obo/" + e["id"]])
                            showed_ids.append("http://purl.obolibrary.org/obo/" + e["id"])
            count_ent += len(ent_list)
            max_ent_num = max(max_ent_num, len(ent_list))
            new_d["entity_list"] = ent_list
            new_d_list.append(new_d)
        with open(f"../../../data/hpo/cr_input_json/{index_type}_{cate}.json", "w") as f_out:
            json.dump(new_d_list, f_out)
            f_out.close()
        print(f"{len(new_d_list)} abstracts ({count_ent} entities) are processed for {cate} set.")
        print(f"One instance contains most at {max_ent_num} entities for {cate} set.")


def get_hpo_span_id_pairs(index_type: str, id_to_index_file: str):
    with open(id_to_index_file, "r") as f_in:
        id_to_index_map = json.load(f_in)
        f_in.close()

    with open(f"../../../data/hpo/ori_json/train_ori.json", "r") as f_in:
        all_d = json.load(f_in)
        f_in.close()

    mention_id_dict = {}
    showed_id = []
    for d_id, d in all_d.items():
        for e in d["entities"]:
            e["id"] = "http://purl.obolibrary.org/obo/" + e["id"]

            if e["name"] not in mention_id_dict.keys():
                mention_id_dict[e["name"]] = []
            if e["id"] not in mention_id_dict[e["name"]]:
                if e["id"] not in id_to_index_map.keys():
                    print(e)
                showed_id.append(e["id"])
                mention_id_dict[e["name"]].append(e["id"])

    with open(f"../../../data/hpo/ori_json/hpo_ontology.json", "r") as f_in:
        id_to_ent_map = json.load(f_in)
        f_in.close()

    name_id_dict = {}
    synonym_id_dict = {}
    count_c_name = 0
    for e_id, e in id_to_ent_map.items():
        if e_id not in showed_id:
            continue
        count_c_name += 1
        if e[0] not in name_id_dict.keys():
            name_id_dict[e[0]] = [e_id]
        else:
            if e_id not in name_id_dict[e[0]]:
                name_id_dict[e[0]].append(e_id)
        for s in e[-1]:
            if s not in synonym_id_dict.keys():
                synonym_id_dict[s] = [e_id]
            else:
                if e_id not in synonym_id_dict[s]:
                    synonym_id_dict[s].append(e_id)

    # Mention - id pairs
    new_d_list = []
    for mention, id_list in mention_id_dict.items():

        ssid = ""
        for e_id in id_list:
            ssid += "-".join([str(x) for x in id_to_index_map[e_id]])
            ssid += ";"
        new_d = {
            "ent_id": "|".join(id_list),
            "label": mention,
            "ssid": ssid
        }
        # if len(id_list) > 1:
        #     print(new_d)
        new_d_list.append(new_d)

    with open(f"../../../data/hpo/cr_input_json/{index_type}_train_mention_to_id.json", "w") as f_out:
        json.dump(new_d_list, f_out)
        f_out.close()
    print(
        f"{len(new_d_list)} mention - ssid pairs ({count_c_name} entities) are processed for cr task model training.")

    # Name - id pairs
    new_d_list = []
    for name, id_list in name_id_dict.items():

        ssid = ""
        for e_id in id_list:
            ssid += "-".join([str(x) for x in id_to_index_map[e_id]])
            ssid += ";"
        new_d = {
            "ent_id": "|".join(id_list),
            "label": name,
            "ssid": ssid
        }
        if len(id_list) > 1:
            print(new_d)
        new_d_list.append(new_d)

    with open(f"../../../data/hpo/cr_input_json/{index_type}_train_name_to_id.json", "w") as f_out:
        json.dump(new_d_list, f_out)
        f_out.close()
    print(
        f"{len(new_d_list)} name - ssid pairs ({count_c_name} entities) are processed for cr task model training.")

    # Synonym - id pairs
    new_d_list = []
    for name, id_list in synonym_id_dict.items():

        ssid = ""
        for e_id in id_list:
            ssid += "-".join([str(x) for x in id_to_index_map[e_id]])
            ssid += ";"
        new_d = {
            "ent_id": "|".join(id_list),
            "label": name,
            "ssid": ssid
        }
        # if len(id_list) > 1:
        #     print(new_d)
        new_d_list.append(new_d)

    with open(f"../../../data/hpo/cr_input_json/{index_type}_train_synonym_to_id.json", "w") as f_out:
        json.dump(new_d_list, f_out)
        f_out.close()
    print(
        f"{len(new_d_list)} synonym - ssid pairs ({count_c_name} entities) are processed for cr task model training.")


def get_cdr_passage_id_pairs(index_type: str, id_to_index_file: str):
    with open(id_to_index_file, "r") as f_in:
        id_to_index_map = json.load(f_in)
        f_in.close()

    cate_list = ["train", "dev", "test"]
    for cate in cate_list:
        max_ent_num = 0
        with open(f"../../../data/cdr/ori_json/{cate}_ori.json", "r") as f_in:
            all_d = json.load(f_in)
            f_in.close()
        new_d_list = []
        count_ent = 0
        for d_id, d in all_d.items():
            new_d = {
                "doc_key": d_id,
                "description": d["abstract"]
            }
            ent_list = []
            showed_ids = []
            for e in d["entities"]:
                if e["id"] not in showed_ids:
                    if e["id"] not in id_to_index_map.keys() and "|" in e["id"]:
                        for ee_id in e["id"].split("|"):
                            if ee_id not in showed_ids and ee_id != "-1":
                                ent_list.append(id_to_index_map[ee_id])
                                showed_ids.append(ee_id)
                    else:
                        ent_list.append(id_to_index_map[e["id"]])
                        showed_ids.append(e["id"])
            count_ent += len(ent_list)
            max_ent_num = max(max_ent_num, len(ent_list))
            new_d["entity_list"] = ent_list
            new_d_list.append(new_d)
        with open(f"../../../data/cdr/cr_input_json/ssid_{cate}.json", "w") as f_out:
            json.dump(new_d_list, f_out)
            f_out.close()
        print(f"{len(new_d_list)} abstracts ({count_ent} entities) are processed for {cate} set.")
        print(f"One instance contains most at {max_ent_num} entities for {cate} set.")


def get_cdr_span_id_pairs(index_type: str, id_to_index_file: str):
    with open(id_to_index_file, "r") as f_in:
        id_to_index_map = json.load(f_in)
        f_in.close()
    with open(f"../../../data/cdr/ori_json/train_ori.json", "r") as f_in:
        all_d = json.load(f_in)
        f_in.close()

    mention_id_dict = {}
    showed_id = []
    for d_id, d in all_d.items():
        for e in d["entities"]:
            if e["name"] not in mention_id_dict.keys():
                mention_id_dict[e["name"]] = []

            if e["id"] not in id_to_index_map.keys() and "|" in e["id"]:
                for ee_id in e["id"].split("|"):
                    if ee_id != "-1":
                        if ee_id not in mention_id_dict[e["name"]]:
                            mention_id_dict[e["name"]].append(ee_id)
                            showed_id.append(ee_id)
            else:
                if e["id"] not in mention_id_dict[e["name"]]:
                    showed_id.append(e["id"])
                    mention_id_dict[e["name"]].append(e["id"])
    with open(f"../../../data/cdr/ori_json/mesh_ontology.json", "r") as f_in:
        id_to_ent_map = json.load(f_in)
        f_in.close()

    name_id_dict = {}
    synonym_id_dict = {}
    count_c_name = 0
    for e_id, e in id_to_ent_map.items():
        if e_id not in showed_id:
            continue
        count_c_name += 1
        if e[0] not in name_id_dict.keys():
            name_id_dict[e[0]] = [e_id]
        else:
            if e_id not in name_id_dict[e[0]]:
                name_id_dict[e[0]].append(e_id)
        for s in e[-1]:
            if s not in synonym_id_dict.keys():
                synonym_id_dict[s] = [e_id]
            else:
                if e_id not in synonym_id_dict[s]:
                    synonym_id_dict[s].append(e_id)

    # Mention - id pairs
    new_d_list = []
    for mention, id_list in mention_id_dict.items():

        ssid = ""
        for e_id in id_list:
            ssid += "-".join([str(x) for x in id_to_index_map[e_id]])
            ssid += ";"
        new_d = {
            "ent_id": "|".join(id_list),
            "label": mention,
            "ssid": ssid
        }
        # if len(id_list) > 1:
        #     print(new_d)
        new_d_list.append(new_d)

    with open(f"../../../data/cdr/cr_input_json/{index_type}_train_mention_to_id.json", "w") as f_out:
        json.dump(new_d_list, f_out)
        f_out.close()
    print(
        f"{len(new_d_list)} mention - ssid pairs ({count_c_name} entities) are processed for cr task model training.")

    # Name - id pairs
    new_d_list = []
    for name, id_list in name_id_dict.items():

        ssid = ""
        for e_id in id_list:
            ssid += "-".join([str(x) for x in id_to_index_map[e_id]])
            ssid += ";"
        new_d = {
            "ent_id": "|".join(id_list),
            "label": name,
            "ssid": ssid
        }
        if len(id_list) > 1:
            print(new_d)
        new_d_list.append(new_d)

    with open(f"../../../data/cdr/cr_input_json/{index_type}_train_name_to_id.json", "w") as f_out:
        json.dump(new_d_list, f_out)
        f_out.close()
    print(
        f"{len(new_d_list)} name - ssid pairs ({count_c_name} entities) are processed for cr task model training.")

    # Synonym - id pairs
    new_d_list = []
    for name, id_list in synonym_id_dict.items():

        ssid = ""
        for e_id in id_list:
            ssid += "-".join([str(x) for x in id_to_index_map[e_id]])
            ssid += ";"
        new_d = {
            "ent_id": "|".join(id_list),
            "label": name,
            "ssid": ssid
        }
        # if len(id_list) > 1:
        #     print(new_d)
        new_d_list.append(new_d)

    with open(f"../../../data/cdr/cr_input_json/{index_type}_train_synonym_to_id.json", "w") as f_out:
        json.dump(new_d_list, f_out)
        f_out.close()
    print(
        f"{len(new_d_list)} synonym - ssid pairs ({count_c_name} entities) are processed for cr task model training.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="hoip")
    parser.add_argument('--index_type', type=str, default="ssid")
    args, _1 = parser.parse_known_args()

    if args.index_type != "ssid" and args.dataset != "hoip":
        raise ValueError(f"Only HOIP dataset is supported with {args.index_type}.")

    index_type = args.index_type
    if index_type == "ssid":
        id_to_index_file = f"../../../data/{args.dataset}/search_index_related/id_ssid_k10_c10_map.json"
    elif index_type == "randomid":
        id_to_index_file = f"../../../data/{args.dataset}/search_index_related/id_random_id_map.json"
    elif index_type == "ssid_w_hypernym":
        id_to_index_file = f"../../../data/{args.dataset}/search_index_related/id_ssid_w_hypernym_k10_c10_map.json"
    elif index_type == "ontoid":
        id_to_index_file = f"../../../data/{args.dataset}/search_index_related/id_onto_id_map.json"
    else:
        raise ValueError("Index type is not valid.")

    if args.dataset == "hoip":
        get_hoip_passage_id_pairs(index_type=index_type, id_to_index_file=id_to_index_file)
        get_hoip_span_id_pairs(index_type=index_type, id_to_index_file=id_to_index_file)

    elif args.dataset == "hpo":
        get_hpo_passage_id_pairs(index_type=index_type, id_to_index_file=id_to_index_file)
        get_hpo_span_id_pairs(index_type=index_type, id_to_index_file=id_to_index_file)

    elif args.dataset == "cdr":
        get_cdr_passage_id_pairs(index_type=index_type, id_to_index_file=id_to_index_file)
        get_cdr_span_id_pairs(index_type=index_type, id_to_index_file=id_to_index_file)

    else:
        raise ValueError("You should assign the 'dataset' as 'cdr', 'hpo' or 'hoip'.")

