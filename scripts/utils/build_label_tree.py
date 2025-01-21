import json
import os

import torch
import ast
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import Dataset


def get_ent_list_hoip(file_path: str):
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
    # all_label_list = all_label_list[:1200]
    ent_df = pd.DataFrame(
        {"label": [d[0] for d in all_label_list], "entity_id": [d[1] for d in all_label_list]})
    print(f"{len(ent_df['label'].values)} process entities are loaded from ontology.")
    print(f"{count_hoip_process} process entities are loaded as HOIP items.")

    return ent_df


def get_ent_list_cdr(file_path: str):
    with open(file_path, "r") as f_in:
        ontology_list = json.load(f_in)
        f_in.close()
    all_label_list = []
    for o_id, o in ontology_list.items():
        all_label_list.append([o[0], o_id])
        # print(o)
        # break
    # all_label_list = all_label_list[:1200]
    ent_df = pd.DataFrame({"label": [d[0] for d in all_label_list], "entity_id": [d[1] for d in all_label_list]})
    print(f"{len(ent_df['label'].values)} process entities are loaded from ontology.")
    return ent_df


def get_ent_list_hpo(file_path: str):
    with open(file_path, "r") as f_in:
        ontology_dict = json.load(f_in)
        f_in.close()
    all_label_list = []

    print(f"{len(ontology_dict['graphs'][0]['nodes'])} labels are loaded from ontology.")

    for o in ontology_dict["graphs"][0]["nodes"]:
        if "type" in o.keys() and o["type"] == "CLASS":
            all_label_list.append([o["lbl"], o["id"]])

    ent_df = pd.DataFrame({"label": [d[0] for d in all_label_list], "entity_id": [d[1] for d in all_label_list]})
    print(f"{len(ent_df['label'].values)} entities are loaded from ontology.")
    return ent_df


def get_ent_emb(model, tokenizer, ent_list, device):
    with torch.no_grad():
        tokens = {'input_ids': [], 'attention_mask': []}
        idx_list = []
        label_list = []

        for t_id, ent_name in ent_list:
            cur_tokens = tokenizer.encode_plus(ent_name, max_length=128,
                                               truncation=True, padding='do_not_pad',
                                               return_tensors='pt')
            tokens['input_ids'].append(cur_tokens['input_ids'][0])
            # tokens['attention_mask'].append(cur_tokens['attention_mask'][0])
            idx_list.append(t_id)
            label_list.append(ent_name)
        input_data = []
        for idx in tokens["input_ids"]:
            input_data.append({
                "input_ids": idx.tolist(),
            })
        input_data = Dataset.from_list(input_data)

        # del tokens
        def collate_emb_fn(batch):
            input_ids = [f["input_ids"] + [tokenizer.pad_token_id] * (128 - len(f["input_ids"])) for f in batch]
            input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (128 - len(f["input_ids"])) for f in batch]

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.float)
            return input_ids, input_mask

        data_loader = DataLoader(input_data, batch_size=32, shuffle=False,
                                 collate_fn=collate_emb_fn,
                                 drop_last=False)
        all_mean_pooled_embed = []

        sub_p_bar = tqdm(total=len(data_loader))

        for batch in data_loader:
            inputs = {
                'input_ids': batch[0].to(device),
                'attention_mask': batch[1].to(device),
            }

            outputs = model(**inputs, output_hidden_states=True)
            # embeddings = (outputs.last_hidden_state + outputs.hidden_states[0]) / 2
            embeddings = outputs.last_hidden_state

            # To perform this operation, we first resize our attention_mask tensor:
            # attention_mask = tokens['attention_mask']

            attention_mask = batch[1]
            mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float().to(device)
            # print(embeddings.device, mask.device)
            masked_embeddings = embeddings * mask

            summed = torch.sum(masked_embeddings, 1)
            summed_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled_embed = summed / summed_mask
            mean_pooled_embed = torch.nn.functional.normalize(mean_pooled_embed, p=2, dim=1)

            # mean_pooled_embed = embeddings[:, 0]
            all_mean_pooled_embed.extend(mean_pooled_embed.cpu().numpy())
            # print(mean_pooled_embed.size())
            sub_p_bar.update(1)
        sub_p_bar.close()

        print("all:", all_mean_pooled_embed.__len__())
        # del summed, summed_mask
    return idx_list, np.array([e.tolist() for e in all_mean_pooled_embed]), label_list


if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--v_dim', type=int, default=768)
    parser.add_argument('--bert_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--c', type=int, default=10)
    parser.add_argument('--read_emb', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="hpo")
    parser.add_argument('--embedding_model', type=str, default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    args = parser.parse_args()

    # Get the input entity list
    if args.dataset == "hoip":
        args.onto_file_path = "../../data/hoip/ori/hoip_ontology.json"
        ent_df = get_ent_list_hoip(file_path=args.onto_file_path)
    elif args.dataset == "cdr":
        args.onto_file_path = "../../data/cdr/ori_json/mesh_ontology.json"
        ent_df = get_ent_list_cdr(file_path=args.onto_file_path)
    elif args.dataset == "hpo":
        args.onto_file_path = "../../data/hpo/ori/hp.json"
        ent_df = get_ent_list_hpo(file_path=args.onto_file_path)

    args.device = "mps"

    # Get the input entity embedding (by SapBERT)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    model = AutoModel.from_pretrained(args.embedding_model)
    model.to(args.device)

    # Define layer to be used
    if args.read_emb == 0:
        layer = model.config.num_hidden_layers - 1

        idx_list, all_seq_embeddings, label_list = get_ent_emb(model, tokenizer,
                                                               zip(ent_df["entity_id"].tolist(),
                                                                   ent_df["label"].tolist()),
                                                               device=args.device)
        print("Loaded all input embedding:", all_seq_embeddings.shape)

        if not os.path.exists(f"../models/{args.dataset}/label_tree"):
            os.mkdir(f"../models/{args.dataset}/label_tree")

        entity_emb = [idx_list, all_seq_embeddings, label_list]
        with open(
                f'../models/{args.dataset}/label_tree/Entity_NQ_bert_{args.bert_size}_k{args.k}_c{args.c}_seed_{args.seed}.pkl',
                'wb') as f:
            pickle.dump(entity_emb, f)
            f.close()
    elif args.read_emb == 1:

        with open(
                f'../models/{args.dataset}/label_tree/Entity_NQ_bert_{args.bert_size}_k{args.k}_c{args.c}_seed_{args.seed}.pkl',
                'rb') as f:
            entity_emb = pickle.load(f)
            f.close()
            # print(entity_emb[0][0], entity_emb[1][0][:20], entity_emb[2][0])
            all_seq_embeddings = entity_emb[1]
    else:
        raise ValueError("`read_emb` should be set as 0 or 1.")

    # Set the clustering model
    kmeans = KMeans(n_clusters=args.k, max_iter=300, n_init=100, init='k-means++', random_state=args.seed, tol=1e-7)

    mini_kmeans = MiniBatchKMeans(n_clusters=args.k, max_iter=300, n_init=100, init='k-means++', random_state=3,
                                  batch_size=1000, reassignment_ratio=0.01, max_no_improvement=20, tol=1e-7)

    ssid_list = []
    center_list = []

    # First clustering
    print("Running the 1st clustering...")
    labels = mini_kmeans.fit_predict(all_seq_embeddings)
    center_list.append(["", mini_kmeans.cluster_centers_])
    for class_ in labels:
        ssid_list.append([class_])


    def recursively_clustering(temp_id_list):
        if temp_id_list.shape[0] <= args.c:
            if temp_id_list.shape[0] == 1:
                return
            for idx, row_id in enumerate(temp_id_list):
                ssid_list[row_id].append(idx)
            return

        temp_data = np.zeros((temp_id_list.shape[0], args.v_dim))
        for idx, row_id in enumerate(temp_id_list):
            temp_data[idx, :] = all_seq_embeddings[row_id]

        if temp_id_list.shape[0] >= 1e3:
            temp_labels = mini_kmeans.fit_predict(temp_data)
            center_list.append(["-".join([str(e) for e in ssid_list[temp_id_list[0]]]), mini_kmeans.cluster_centers_])
        else:
            temp_labels = kmeans.fit_predict(temp_data)
            center_list.append(["-".join([str(e) for e in ssid_list[temp_id_list[0]]]), kmeans.cluster_centers_])

        for temp_i in range(args.k):
            new_ssid_list = []
            for temp_id, temp_class_ in enumerate(temp_labels):
                if temp_class_ == temp_i:
                    new_ssid_list.append(temp_id_list[temp_id])
                    ssid_list[temp_id_list[temp_id]].append(temp_i)
            recursively_clustering(np.array(new_ssid_list))

        return


    # Recursively clustering
    for i in range(args.k):
        print(f"Running the clustering of the category {i}...")
        temp_ssid_list = []
        for id_, class_ in enumerate(labels):
            if class_ == i:
                temp_ssid_list.append(id_)
        recursively_clustering(np.array(temp_ssid_list))

    onto_id = ent_df["entity_id"].tolist()

    mapping = {}
    for i in range(len(onto_id)):
        mapping[onto_id[i]] = ssid_list[i]

    ent_df["ssid"] = ssid_list

    output_file_path = f'../../data/{args.dataset}/search_index_related/to_ssid_k{args.k}_c{args.c}.csv'
    ent_df.to_csv(output_file_path, sep='\t', index=False)

    with open(
            f'../models/{args.dataset}/label_tree/IDMapping_NQ_bert_{args.bert_size}_k{args.k}_c{args.c}_seed_{args.seed}.pkl',
            'wb') as f:
        pickle.dump(mapping, f)

    with open(
            f'../models/{args.dataset}/label_tree/Center_NQ_bert_{args.bert_size}_k{args.k}_c{args.c}_seed_{args.seed}.pkl',
            'wb') as f:
        pickle.dump(center_list, f)

    # To get id-ssid map for further processing

    ssid_label_map = {}
    to_ssid_df = pd.read_csv(output_file_path, sep='\t')

    id_to_ssid_map = {}

    for e_id, ent_id, label in zip(to_ssid_df["ssid"], to_ssid_df["entity_id"], to_ssid_df["label"]):
        e_id = ast.literal_eval(e_id)
        ssid_label_map["-".join([str(c) for c in e_id])] = label
        id_to_ssid_map[ent_id] = e_id

    with open(f'../../data/{args.dataset}/search_index_related/id_ssid_k{args.k}_c{args.c}_map.json', "w") as f_out:
        json.dump(id_to_ssid_map, f_out)
        f_out.close()

    print("Finished.")
