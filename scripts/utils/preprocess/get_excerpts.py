import argparse
import json
import spacy
import tqdm
from spacy.lang.en import English
from spacy.symbols import nsubj, VERB, NOUN, PROPN


def get_spans(input_file_path: str, output_file_path: str, input_type):
    with open(input_file_path, "r") as f_in:
        all_d = json.load(f_in)
        f_in.close()

    depandency_parser = spacy.load("en_core_web_trf")
    depandency_parser.add_pipe("merge_noun_chunks")

    # if input_type == "passage":
    #     sentence_tokenizer = English()
    #     sentence_tokenizer.add_pipe("sentencizer")
    count_spans = 0
    count_sents = 0

    new_d = {}

    pbar = tqdm.tqdm(total=len(all_d.keys()))

    for d_id, d in all_d.items():
        if input_type == "passage":
            new_d[d_id] = {"passage": d["passage"]}
            d_sents = []
            cur_c_id = 0
            sent_list = d["passage"].split(". ")
            for s in sent_list:
                s = s + "."
                cur_c_id += len(s)
                if cur_c_id < len(d["passage"]) and d["passage"][cur_c_id - len(s)].isupper():
                    d_sents.append(s)
                elif cur_c_id == len(d["passage"]) + 2:
                    d_sents.append(s[:-2] + " ")
                else:
                    if len(d_sents) > 0:
                        d_sents[-1] += s
                    else:
                        # print(s)
                        d_sents.append(s)
        elif input_type == "claim":
            new_d[d_id] = {"passage": d["passage"]}
            d_sents = d["g_claim"]
        else:
            raise ValueError("Only support `passage` or `claim`.")

        new_d[d_id]["excerpt"] = []

        # print(d_sents)
        count_sents += len(d_sents)

        for c_id, c in enumerate(d_sents):
            # print(c)
            p_doc = depandency_parser(c)

            p_list = []
            cur_head = None

            # for token in p_doc:
            #     # if token.tag_ == "VBZ" or token.tag_.startswith("V"):
            #     print(
            #         f"""TOKEN: {token.text} | {token.tag_ = } | {token.pos_ = } | {token.head.text = } | {token.dep_ = }"""
            #     )
            #     p_list.append([token.text, token.tag_, token.head.text, token.dep_])

            # root = [token for token in p_doc if token.head == token][0]
            verb_span = []
            verb_root = [token for token in p_doc if token.pos == VERB or token.head == token]
            for root in verb_root:
                # print("||root", root.text)
                target = []
                if len(list(root.rights)) == 0:
                    continue
                subject = list(root.rights)[0]
                for descendant in subject.subtree:
                    assert subject is descendant or subject.is_ancestor(descendant)
                    if descendant.pos != NOUN:
                        continue
                    real_ancestor = []
                    for ancestor in descendant.ancestors:
                        if ancestor.text == root.text:
                            if root.text not in ["is", "are", "am"]:
                                real_ancestor.append(ancestor)
                            break

                        real_ancestor.append(ancestor)
                    if len(real_ancestor) == 0:
                        continue

                    min_idx = min(ancestor.idx for ancestor in real_ancestor)
                    min_idx = min(min_idx, descendant.idx)
                    max_idx = max(ancestor.idx + len(ancestor.text) for ancestor in real_ancestor)
                    max_idx = max(max_idx, descendant.idx + len(descendant.text))

                    target = (min_idx, max_idx)
                if len(target) == 2:
                    # print(c[target[0]:target[1]])
                    verb_span.append(target)

            for root in verb_root:

                target = []
                if len(list(root.lefts)) == 0:
                    continue

                subject = list(root.lefts)[0]

                for descendant in subject.subtree:
                    assert subject is descendant or subject.is_ancestor(descendant)
                    if descendant.pos != NOUN:
                        continue
                    real_ancestor = []
                    for ancestor in descendant.ancestors:
                        if ancestor.text == root.text:
                            if root.text not in ["is", "are", "am"]:
                                real_ancestor.append(ancestor)
                            break
                        real_ancestor.append(ancestor)
                    if len(real_ancestor) == 0:
                        continue

                    min_idx = min(ancestor.idx for ancestor in real_ancestor)
                    min_idx = min(min_idx, descendant.idx)
                    max_idx = max(ancestor.idx + len(ancestor.text) for ancestor in real_ancestor)
                    max_idx = max(max_idx, descendant.idx + len(descendant.text))

                    target = (min_idx, max_idx)
                if len(target) == 2:
                    # print(c[target[0]:target[1]])
                    verb_span.append(target)

            noun_span = []

            noun_root = [token for token in p_doc if token.pos == NOUN]
            for root in noun_root:
                noun_span.append((root.idx, root.idx + len(root.text)))
                target = []
                if len(list(root.rights)) == 0:
                    continue
                subject = list(root.rights)[0]
                for descendant in subject.subtree:
                    assert subject is descendant or subject.is_ancestor(descendant)
                    if descendant.pos != NOUN:
                        continue
                    real_ancestor = []
                    for ancestor in descendant.ancestors:
                        if ancestor.text == root.text:
                            if ancestor.pos == NOUN:
                                real_ancestor.append(ancestor)
                            break

                        real_ancestor.append(ancestor)
                    if len(real_ancestor) == 0:
                        continue
                    min_idx = min(ancestor.idx for ancestor in real_ancestor)
                    min_idx = min(min_idx, descendant.idx)
                    max_idx = max(ancestor.idx + len(ancestor.text) for ancestor in real_ancestor)
                    max_idx = max(max_idx, descendant.idx + len(descendant.text))

                    target = (min_idx, max_idx)
                    break
                if len(target) == 2:
                    # print(c[target[0]:target[1]])
                    noun_span.append(target)
            for root in noun_root:
                target = []
                if len(list(root.lefts)) == 0:
                    continue
                subject = list(root.lefts)[0]
                for descendant in subject.subtree:
                    assert subject is descendant or subject.is_ancestor(descendant)
                    if descendant.pos != NOUN:
                        continue
                    real_ancestor = []
                    for ancestor in descendant.ancestors:
                        if ancestor.text == root.text:
                            if ancestor.pos == NOUN:
                                real_ancestor.append(ancestor)
                            break

                        real_ancestor.append(ancestor)
                    if len(real_ancestor) == 0:
                        continue
                    min_idx = min(ancestor.idx for ancestor in real_ancestor)
                    min_idx = min(min_idx, descendant.idx)
                    max_idx = max(ancestor.idx + len(ancestor.text) for ancestor in real_ancestor)
                    max_idx = max(max_idx, descendant.idx + len(descendant.text))

                    target = (min_idx, max_idx)
                    break
                if len(target) == 2:
                    # print(c[target[0]:target[1]])
                    noun_span.append(target)

            b_noun_span = []

            cur_s_id = -1
            cur_e_id = -1
            for t_id, token in enumerate(p_doc):
                if (token.pos == NOUN or token.pos == PROPN or token.text == "and") and cur_s_id != -1:
                    cur_e_id = t_id
                elif (token.pos == NOUN or token.pos == PROPN) and cur_s_id == -1:
                    cur_s_id = t_id
                else:
                    if cur_e_id != -1 and cur_s_id != -1:
                        b_noun_span.append((p_doc[cur_s_id].idx, p_doc[cur_e_id].idx + len(p_doc[cur_e_id].text)))
                        cur_e_id = -1
                        cur_s_id = -1
                        # print(c[noun_span[-1][0]:noun_span[-1][1]])
                    else:
                        cur_e_id = -1
                        cur_s_id = -1

            tmp = []
            for k in noun_span:
                in_flag = False
                for b in b_noun_span:
                    if k[0] >= b[0] and k[1] <= b[1]:
                        in_flag = True
                        break

                for b in noun_span:
                    if (str(c[k[0]:k[1]]) + " of ") in c[b[0]:b[1]]:
                        in_flag = True
                        break

                    if (str(c[k[0]:k[1]]) + " with ") in c[b[0]:b[1]]:
                        in_flag = True
                        break
                if not in_flag:
                    tmp.append(k)
            noun_span = tmp

            all_span = noun_span + b_noun_span + verb_span
            # all_span = list(set(all_span))

            # for k in noun_span + b_noun_span + verb_span:
            #     print(c[k[0]:k[1]])

            # new_d[d_id]["Spans][c_id] = list(set([[k[0], k[1], c[k[0]:k[1]]] for k in all_span]))
            # new_d[d_id]["Spans][c_id] = list(set([c[k[0]:k[1]] for k in all_span]))
            new_d[d_id]["excerpt"].extend(list(set([c[k[0]:k[1]] for k in all_span])))

            # print("=" * 30)
        new_d[d_id]["excerpt"] = sorted(list(set(new_d[d_id]["excerpt"])))
        count_spans += len(new_d[d_id]["excerpt"])
        pbar.update(1)
        # break
        # if count_sents > 100:
        #     break
    pbar.close()
    with open(output_file_path, "w") as f_out:
        json.dump(new_d, f_out)
        f_out.close()

    print(f"{count_spans} spans are generated from {count_sents} sentences/claims from {len(new_d)} passages.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="hoip")
    parser.add_argument('--input_type', type=str, default="passage", help="Could be set as `passage` or `claim`.")
    parser.add_argument('--input_file_path', type=str, default="test_ori.json")
    parser.add_argument('--output_file_path', type=str, default="test_ori_excerpt.json")

    args, _1 = parser.parse_known_args()

    input_file_path = f"../../../data/{args.dataset}/ori_json/{args.input_file_path}"
    output_file_path = f"../../../data/{args.dataset}/ori_json/{args.output_file_path}"

    get_spans(input_file_path=input_file_path, output_file_path=output_file_path, input_type=args.input_type)
