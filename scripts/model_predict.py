import argparse
import os

import torch
import json
import pandas as pd
import ast

from transformers import BartTokenizerFast, BartConfig, BartModel
from networks.macoir import MACOIR
from torch.utils.data import DataLoader
from utils.reader import load_text


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


def do_pred(model, dev_loader, device, log_file_path, model_card_name, num_return_sequences=10):
    predictions = []
    ent_ids = []
    ent_texts = []
    ent_types = []
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
        for src_ids, src_mask, doc_ids, doc_texts, doc_types in dev_loader:

            inputs = {
                'input_ids': src_ids.to(device),
                'attention_mask': src_mask.to(device),
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

            ent_ids.extend(doc_ids)
            ent_texts.extend(doc_texts)
            ent_types.extend(doc_types)

    res = []
    for p, ent_id, ent_text, ent_type in zip(predictions, ent_ids, ent_texts, ent_types):
        res.append([ent_id, ent_text, ent_type, p])
    with open(log_file_path + f'/log_{model_card_name}_pred.json', "w") as f_out:
        json.dump(res, f_out)
        f_out.close()


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

    print(f"Loaded model_{args.model_card_name}.pt for subsequent prediction...")

    if args.prediction:
        load_checkpoint(args.model_dir + f"/model_{args.model_card_name}.pt", model, device=args.device)
        model.eval()
        # Set model
        model.to(args.device)
        print("Now the prediction is running...")


        def collate_pred_fn(batch):
            src_ids = [f["src_ids"] + [tokenizer.pad_token_id] * (max_src_len - len(f["src_ids"])) for f in batch]
            src_mask = [[1.0] * len(f["src_ids"]) + [0.0] * (max_src_len - len(f["src_ids"])) for f in batch]

            doc_ids = [f["doc_key"] for f in batch]
            doc_types = [f["doc_type"] for f in batch]
            ent_texts = [f["text"] for f in batch]

            src_ids = torch.tensor(src_ids, dtype=torch.long)
            src_mask = torch.tensor(src_mask, dtype=torch.float)
            return src_ids, src_mask, doc_ids, ent_texts, doc_types


        all_data, max_src_len, max_tgt_len = load_text(args.dataset, args.pred_path, tokenizer,
                                                       max_seq_length=1024)

        dev_loader = DataLoader(all_data, batch_size=args.per_device_train_batch_size,
                                shuffle=False,
                                collate_fn=collate_pred_fn, drop_last=False)
        do_pred(model=model, dev_loader=dev_loader, log_file_path=args.log_dir, device=args.device,
                model_card_name=args.model_card_name, num_return_sequences=args.num_return_sequences)

    print('Finished.')
