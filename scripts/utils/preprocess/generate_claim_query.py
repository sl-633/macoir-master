import argparse
import json
import os

import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="hoip", type=str, help='Can be set as `cdr`, `hpo` or `hoip`.')
    parser.add_argument('--passage_file', default="test_ori.json", type=str)
    args, unparsed = parser.parse_known_args()

    config_dict = {
        "transformer_id": "meta-llama/Meta-Llama-3-8B-Instruct"
    }
    args.__dict__.update(config_dict)

    args.output_dir = f"../../../data/{args.dataset}/generated_files"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.device = "cuda" if torch.cuda.is_available() else "mps"
    args.num_gpus = torch.cuda.device_count()

    print("num_gpus: ", args.num_gpus, "; device: ", args.device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    huggingface_hub.login(token="")

    tokenizer = AutoTokenizer.from_pretrained(args.transformer_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.transformer_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # load_in_4bit=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    # model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with open(f"../../../data/{args.dataset}/{args.passage_file}", "r") as f_in:
        ori_data = json.load(f_in)
        f_in.close()
    # ori_data = ori_data[4300:]

    all_inst = []
    p_bar = tqdm(total=len(ori_data.keys()))
    for d, c in ori_data.items():
        messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant in the biomedical domain."}, ]

        messages.append({"role": "user",
                         "content": "Please breakdown the following input into a set of small, independent claims (make sure not to add any information), and return the output as a jsonl, where each line is {\"claim\":[CLAIM], \"score\":[CONF]}. The confidence score [CONF] should represent your confidence in the claim, where a 1 is obvious facts and results like ‘The earth is round’ and ‘1+1=2’. A 0 is for claims that are very obscure or difficult for anyone to know, like the birthdays of non-notable people. If the input is short, it is fine to only return 1 claim. Directly return the jsonl with no explanation or other formatting. " + f"The input is: \"{c['passage']}\""})
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        inst = {
            "doc_key": d,
            "generated_answer": generate_per_inst(model, terminators, input_ids),
        }
        all_inst.append(inst)
        p_bar.update(1)

        if len(all_inst) % 100 == 0:
            with open(args.output_dir + f"/generated_claim_{args.passage_file}", "a") as file_out:
                json.dump(all_inst, file_out)
                all_inst = []
                file_out.close()
        # break

    p_bar.close()

    with open(args.output_dir + f"/generated_claim_{args.passage_file}", "a") as file_out:
        json.dump(all_inst, file_out)
        file_out.close()

    print("Finished.")
