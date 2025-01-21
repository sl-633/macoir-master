import json
import os

import ast
import random

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import pickle
import argparse


def get_ent_list(file_path: str):
    with open(file_path, "r") as f_in:
        ontology_list = json.load(f_in)
        f_in.close()
    all_label_list = []
    print(f"{len(ontology_list)} labels are loaded from ontology.")
    # hoip_label = []
    count_hoip_process = 0
    for o in ontology_list:
        # if "HOIP" in o["Class ID"]:
        #     hoip_label.append(o["Class ID"].split("_")[0])

        if "http://purl.obolibrary.org/obo/GO_" not in o[
            "Class ID"] and "http://purl.bioontology.org/ontology/HOIP/HOIP_" not in \
                o["Class ID"]:
            continue
        if "http://purl.obolibrary.org/obo/GO_" in o["Class ID"] and ".6436" not in o["TreeNumbers"][
            0] and ".1396" not in o["TreeNumbers"][0]:
            continue
        if "http://purl.bioontology.org/ontology/HOIP/HOIP_" in o["Class ID"] and ".52802" not in o["TreeNumbers"][0]:
            continue
        for p_l in o["Preferred Labels"]:
            if p_l.startswith("GO_") or p_l.startswith("HOIP_"):
                continue

            all_label_list.append([p_l, o["Class ID"]])
            if "http://purl.bioontology.org/ontology/HOIP/HOIP_" in o["Class ID"]:
                count_hoip_process += 1
            break
        # print(o)
        # break
    # all_label_list = all_label_list[:1200]
    ent_df = pd.DataFrame(
        {"label": [d[0] for d in all_label_list], "entity_id": [d[1] for d in all_label_list]})
    print(f"{len(ent_df['label'].values)} process entities are loaded from ontology.")
    print(f"{count_hoip_process} process entities are loaded as HOIP items.")
    # hoip_label = list(set(hoip_label))
    # print(hoip_label)

    return ent_df


if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="hoip")
    parser.add_argument('--onto_file_path', type=str, default="../../data/hoip/ori/hoip_ontology.json")
    args = parser.parse_args()

    # Get the input entity list
    if args.dataset == "hoip":
        ent_df = get_ent_list(file_path=args.onto_file_path)
    else:
        raise ValueError("We only experiment non-ssid on HOIP dataset, please set `dataset` as `hoip`.")

    label_list = ent_df["label"].tolist()
    onto_id_list = ent_df["entity_id"].tolist()

    # Random ID

    mapping = {}
    random_candidate = [i for i in range(len(onto_id_list))]
    random.shuffle(random_candidate)
    for i in range(len(onto_id_list)):
        mapping[onto_id_list[i]] = random_candidate[i]

    ent_df["ssid"] = random_candidate

    output_file_path = f'../../data/{args.dataset}/search_index_related/to_random_id.csv'
    ent_df.to_csv(output_file_path, sep='\t', index=False)

    # To get id-randon id map for further processing

    with open(f'../../data/{args.dataset}/search_index_related/id_random_id_map.json', "w") as f_out:
        json.dump(mapping, f_out)
        f_out.close()

    # Onto ID

    mapping = {}
    name_list = []
    for i in range(len(onto_id_list)):
        name = onto_id_list[i].split("/")[-1]
        mapping[onto_id_list[i]] = name
        name_list.append(name)

    ent_df["ssid"] = name_list

    output_file_path = f'../../data/{args.dataset}/search_index_related/to_onto_id.csv'
    ent_df.to_csv(output_file_path, sep='\t', index=False)

    # To get id-randon id map for further processing

    with open(f'../../data/{args.dataset}/search_index_related/id_onto_id_map.json', "w") as f_out:
        json.dump(mapping, f_out)
        f_out.close()

    print("Finished.")
