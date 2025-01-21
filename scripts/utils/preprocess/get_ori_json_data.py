import argparse
import json
import os
import random


def process_hoip_split_data():
    train_data_path = "../../../data/hoip/ori/train.json"
    dev_data_path = "../../../data/hoip/ori/dev.json"
    test_data_path = "../../../data/hoip/ori/test.json"

    all_entity = {}

    for cate, data_path in zip(["train", "dev", "test"], [train_data_path, dev_data_path, test_data_path]):
        all_data = {}
        with open(data_path, "r") as f_in:
            all_d = json.load(f_in)
            f_in.close()
        for d in all_d:

            ents = []
            ent_id_list = []
            rels = []
            for e in d["triples"]:
                ents.append({"id": e["head_entity"]["id"], "name": e["head_entity"]["name"]})
                ents.append({"id": e["tail_entity"]["id"], "name": e["tail_entity"]["name"]})
                ent_id_list.append(e["head_entity"]["id"])
                ent_id_list.append(e["tail_entity"]["id"])

                rels.append(
                    {"subj": e["head_entity"]["id"], "rel_id": e["relation"]["id"], "rel_name": e["relation"]["name"],
                     "obj": e["tail_entity"]["id"]})

            ent_id_list = list(set(ent_id_list))
            new_ents = []
            for e_id in ent_id_list:
                for e in ents:
                    if e["id"] == e_id:
                        new_ents.append(e)
                        break

            all_data[d["doc_key"]] = {
                "passage": " ".join(d["sentences"]),
                "entities": new_ents,
                "relations": rels
            }

            for e in new_ents:
                if e["id"] in all_entity.keys() and e["name"] not in all_entity[e["id"]]["name"]:
                    all_entity[e["id"]]["name"].append(e["name"])
                else:
                    all_entity[e["id"]] = {"name": [e["name"]]}

            # break

        with open(f"../../../data/hoip/ori_json/{cate}_ori.json", "w") as f_out:
            json.dump(all_data, f_out)
            f_out.close()

    with open(f"../../../data/hoip/ori_json/hoip_in_dataset_entities.json", "w") as f_out:
        json.dump(all_entity, f_out)
        f_out.close()


def process_hoip_all():
    with open("../../../data/hoip/ori/hoip_ontology.json", "r") as f_in:
        all_entities = json.load(f_in)
        f_in.close()

    with open("../../../data/hoip/ori/process_concept_id_list.json", "r") as f_in:
        all_ids = json.load(f_in)
        f_in.close()
    all_data = {}

    for inst in all_entities:
        c_name = inst["Preferred Labels"][0]
        c_id = inst["Class ID"]
        c_des = inst["Definitions"]
        c_trees = inst["TreeNumbers"]
        c_synset = inst["Synonyms"]
        if len(inst["Preferred Labels"]) > 1:
            c_synset = c_synset + inst["Preferred Labels"][1:]

        if type(c_name) == str and c_id in all_ids:
            all_data[c_id] = [c_name, c_des, c_trees, c_synset]
        else:
            continue

    with open(f"../../../data/hoip/ori_json/hoip_ontology.json", "w") as f_out:
        json.dump(all_data, f_out)
        f_out.close()
    print(f"{len(all_data)} process concepts are processed from HOIP Ontology.")


def process_cdr_split_data():
    train_data_path = "../../../data/cdr/ori/CDR_TrainingSet.PubTator.txt"
    dev_data_path = "../../../data/cdr/ori/CDR_DevelopmentSet.PubTator.txt"
    test_data_path = "../../../data/cdr/ori/CDR_TestSet.PubTator.txt"

    all_entity = {}

    for cate, data_path in zip(["train", "dev", "test"], [train_data_path, dev_data_path, test_data_path]):
        all_inst = []
        with open(data_path, "r") as f_in:
            all_lines = f_in.readlines()
            f_in.close()
        cur_inst = []
        for line in all_lines:
            if len(line.strip()) == 0:
                if len(cur_inst) > 0:
                    all_inst.append(cur_inst)
                    cur_inst = []
            else:
                cur_inst.append(line.strip())
        if len(cur_inst) > 0:
            all_inst.append(cur_inst)
        all_data = {}
        for inst in all_inst:
            assert "|t|" in inst[0]
            d_id, _, t = inst[0].strip().split("|")
            assert "|a|" in inst[1]
            _1, _2, a = inst[1].strip().split("|")
            assert d_id == _1
            e_list = []
            r_list = []

            for i in inst[2:]:
                ele = i.strip().split("\t")
                assert ele[0] == d_id
                if ele[1] != "CID":
                    e_list.append(i.strip().split("\t")[1:])
                else:
                    r_list.append(i.strip().split("\t")[2:])
            e_name_list = []
            new_e_list = []
            for e in e_list:
                # if e[4] == "-1" and e[2] not in e_name_list:
                if e[4] == "-1":
                    # new_e_list.append({"id": "Other", "name": e[2], "type": e[3]})
                    # e_name_list.append(e[2])
                    continue
                elif e[2] not in e_name_list:
                    new_e_list.append({"id": e[4], "name": e[2], "type": e[3]})
                    e_name_list.append(e[2])
                # else:
                #     new_e_list.append({"id": e[4], "name": e[2], "type": e[3]})

            # print(d_id, t, a, e_list, r_list)
            all_data[d_id] = {
                "title": t,
                "abstract": a,
                "entities": new_e_list,
                "relations": r_list
            }
            for e in e_list:
                if e[4] == "-1":
                    if "other" not in all_entity.keys():
                        all_entity["other"] = {"Chemical": [], "Disease": []}
                    all_entity["other"][e[3]].append(e[2])
                    continue
                if e[4] in all_entity.keys():
                    all_entity[e[4]]["name"].append(e[2])
                else:
                    # if len(e) < 6:
                    #     print(e)
                    all_entity[e[4]] = {"type": e[3], "name": [e[2]]}
        with open(f"../../../data/cdr/ori_json/{cate}_ori.json", "w") as f_out:
            json.dump(all_data, f_out)
            f_out.close()
    for d in all_entity.keys():
        if d != "other":
            all_entity[d]["name"] = list(set(all_entity[d]["name"]))
    all_entity["other"]["Chemical"] = list(set(all_entity["other"]["Chemical"]))
    all_entity["other"]["Disease"] = list(set(all_entity["other"]["Disease"]))

    with open(f"../../../data/cdr/ori_json/cdr_in_dataset_entities.json", "w") as f_out:
        json.dump(all_entity, f_out)
        f_out.close()


def process_mesh_all():
    with open("../../../data/cdr/ori/d2015.bin", "r") as f_in:
        all_lines = f_in.readlines()
        f_in.close()
    all_inst = []

    cur_inst = []
    for line in all_lines:
        if line.strip() == "*NEWRECORD":
            if len(cur_inst) > 0:
                all_inst.append(cur_inst)
                cur_inst = []
        else:
            cur_inst.append(line.strip())
    if len(cur_inst) > 0:
        all_inst.append(cur_inst)

    all_data = {}

    for inst in all_inst:
        c_name = None
        c_id = None
        c_des = None
        c_trees = []
        c_synset = []
        for c in inst:
            if c.startswith("MH = "):
                c_name = c[5:]
            if c.startswith("UI = "):
                c_id = c[5:]
            if c.startswith("MS = "):
                c_des = c[5:]
            if c.startswith("MN = "):
                c_trees.append(c[5:])
            if c.startswith("SY = "):
                c_synset.append(c[5:].split("|")[0])
        if type(c_name) == str:
            c_synset = list(set(c_synset))
            all_data[c_id] = [c_name, c_des, c_trees, c_synset]
        else:
            print(c_id, c_name)

    all_inst = []

    with open("../../../data/cdr/ori/c2015.bin", "r") as f_in:
        all_lines = f_in.readlines()
        f_in.close()

    cur_inst = []
    for line in all_lines:
        if line.strip() == "*NEWRECORD":
            if len(cur_inst) > 0:
                all_inst.append(cur_inst)
                cur_inst = []
        else:
            cur_inst.append(line.strip())
    if len(cur_inst) > 0:
        all_inst.append(cur_inst)

    for inst in all_inst:
        c_name = None
        c_id = None
        c_des = None
        c_trees = []
        c_synset = []
        for c in inst:
            if c.startswith("NM = "):
                c_name = c[5:]
            if c.startswith("UI = "):
                c_id = c[5:]
            if c.startswith("NO = "):
                c_des = c[5:]
            if c.startswith("HM = "):
                c_trees.append(c[5:])
            if c.startswith("SY = "):
                c_synset.append(c[5:].split("|")[0])
        if type(c_name) == str:
            c_synset = list(set(c_synset))
            all_data[c_id] = [c_name, c_des, c_trees, c_synset]
        else:
            print(c_id, c_name)

    del all_inst

    with open(f"../../../data/cdr/ori_json/mesh_ontology.json", "w") as f_out:
        json.dump(all_data, f_out)
        f_out.close()
    print(f"{len(all_data)} entities are processed from MeSH Ontology.")


new_hpo_map = {"HP_0000404": "HP_0000365", "HP_0001390": "HP_0002804", "HP_0002004": "HP_0001999",
               "HP_0002260": "HP_0001999", "HP_0006746": "HP_0001067", "HP_0001617": "HP_0001344",
               "HP_0007319": "HP_0002011", "HP_0003822": "HP_0003812", "HP_0003813": "HP_0003812",
               "HP_0003008": "HP_0002664", "HP_0000792": "HP_0012210", "HP_0000181": "HP_0000154",
               "HP_0003576": "HP_0003593", "HP_0006741": "HP_0002664", "HP_0003815": "HP_0003812",
               "HP_0000209": "HP_0000277", "HP_0002276": "HP_0002311", "HP_0001420": "HP_0003745",
               "HP_0002258": "HP_0000248", "HP_0005188": "HP_0002804", "HP_0008680": "HP_0000104",
               "HP_0007106": "HP_0001263", "HP_0004149": "HP_0003819", "HP_0003664": "HP_0003674",
               "HP_0006096": "HP_0009473", "HP_0001509": "HP_0004322", "HP_0001275": "HP_0001250",
               "HP_0001432": "HP_0003819", "HP_0002391": "HP_0001250", "HP_0007995": "HP_0000589",
               "HP_0004741": "HP_0000089", "HP_0004675": "HP_0001999", "HP_0007669": "HP_0007678",
               "HP_0008657": "HP_0000147", "HP_0007170": "HP_0000750", "HP_0000715": "HP_0000708",
               "HP_0006540": "HP_0010959", "HP_0000156": "HP_0000218", "HP_0003660": "HP_0003577",
               "HP_0002008": "HP_0010628", "HP_0000241": "HP_0005484", "HP_0007746": "HP_0007894",
               "HP_0002116": "HP_0000750", "HP_0006860": "HP_0000496", "HP_0002255": "HP_0002573",
               "HP_0002051": "HP_0000303", "HP_0006996": "HP_0006989", "HP_0000521": "HP_0000632",
               "HP_0010240": "HP_0005819", "HP_0003590": "HP_0003674", "HP_0001759": "HP_0004452",
               "HP_0001255": "HP_0001263"}


def process_hpo_split_data():
    text_data_path = "../../../data/hpo/ori/Text"
    anno_data_path = "../../../data/hpo/ori/Annotations"

    all_data_name = os.listdir(anno_data_path)
    random.shuffle(all_data_name)
    train_d = all_data_name[:182]
    dev_d = all_data_name[182:205]
    test_d = all_data_name[205:]

    all_entity = {}

    for cate, data_list in zip(["train", "dev", "test"], [train_d, dev_d, test_d]):
        all_data = {}
        for d_name in data_list:
            with open(text_data_path + "/" + d_name, "r") as f_in:
                text_d = f_in.readlines()[0]
                f_in.close()
            with open(anno_data_path + "/" + d_name, "r") as f_in:
                anno_d = []
                lines = f_in.readlines()
                for line in lines:
                    _1, _2 = line.strip().split("\t")
                    e_id, e_name = _2.split(" | ")
                    if e_id in new_hpo_map.keys():
                        e_id = new_hpo_map[e_id]
                    e_start, e_end = _1[1:-1].split("::")
                    anno_d.append([e_start, e_end, e_id, e_name])
                f_in.close()

            all_data[d_name] = {
                "abstract": text_d,
                "entities": [{"id": e[2], "name": e[3], "start": e[0], "end": e[1]} for e in anno_d]
            }

            for e in anno_d:
                if e[2] in all_entity.keys() and e[3] not in all_entity[e[2]]["name"]:
                    all_entity[e[2]]["name"].append(e[3])
                else:
                    all_entity[e[2]] = {"name": [e[3]]}

            # break

        with open(f"../../../data/hpo/ori_json/{cate}_ori.json", "w") as f_out:
            json.dump(all_data, f_out)
            f_out.close()

    with open(f"../../../data/hpo/ori/hpo_in_dataset_entities.json", "w") as f_out:
        json.dump(all_entity, f_out)
        f_out.close()


def process_hpo_all():
    with open("../../../data/hpo/ori/hp.json", "r") as f_in:
        all_entities = json.load(f_in)["graphs"][0]["nodes"]
        f_in.close()
    all_data = {}
    for o in all_entities:
        if "type" in o.keys() and o["type"] == "CLASS":
            c_name = o["lbl"]
            c_id = o["id"]
            c_des = o["meta"]["definition"]["val"] if "meta" in o.keys() and "definition" in o["meta"].keys() else ""
            c_trees = []
            c_synset = []
            if "meta" in o.keys() and "synonyms" in o["meta"].keys():
                for c in o["meta"]["synonyms"]:
                    c_synset.append(c["val"])

            all_data[c_id] = [c_name, c_des, c_trees, c_synset]

    with open(f"../../../data/hpo/ori_json/hpo_ontology.json", "w") as f_out:
        json.dump(all_data, f_out)
        f_out.close()
    print(f"{len(all_data)} entities are processed from HPO Ontology.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="hoip")
    args, _1 = parser.parse_known_args()

    if args.dataset == "hoip":
        process_hoip_split_data()
        process_hoip_all()

    elif args.dataset == "cdr":
        process_cdr_split_data()
        process_mesh_all()

    elif args.dataset == "hpo":
        process_hpo_split_data()
        process_hpo_all()
    else:
        raise ValueError("You should assign the 'dataset' as 'cdr', 'hpo' or 'hoip'.")
