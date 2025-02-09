import argparse
import os
import pandas as pd
import ast

from transformers import BartTokenizerFast, BartConfig, BartModel
from networks.macoir import MACOIR
import torch
import json

from tqdm import tqdm
import torch.optim as optim
from datasets import Dataset
from torch.utils.data import DataLoader
from utils.reader import load_passage_pairs, load_span_pairs


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


def compute_metrics(predicted_ids, labels):
    predicted_res = []
    for pp in predicted_ids:
        all_spans = []
        cur_span = []
        for p in pp:
            if p == 131 or p == 2:
                if len(cur_span) != 0:
                    all_spans.append(cur_span)
                    cur_span = []
            elif p == 1:
                break
            else:
                cur_span.append(p)
        if len(cur_span) != 0:
            all_spans.append(cur_span)
        deduplicate_spans = []
        for s in all_spans:
            if s not in deduplicate_spans:
                deduplicate_spans.append(s)
        predicted_res.append(deduplicate_spans)

    gold_res = []
    for pp in labels:
        all_spans = []
        cur_span = []
        for p in pp:
            if p == 131 or p == 2:
                if len(cur_span) != 0:
                    all_spans.append(cur_span)
                    cur_span = []
            elif p == 1:
                break
            else:
                cur_span.append(p)
        if len(cur_span) != 0:
            all_spans.append(cur_span)
        gold_res.append(all_spans)

    tp, fp, fn = 0., 0., 0.
    for p, g in zip(predicted_res, gold_res):
        cur_tp = len([e for e in p if e in g])
        tp += cur_tp
        fp += len([e for e in p if e not in g])
        fn += len([e for e in g if e not in p])

    ent_pre = tp / (tp + fp) if tp + fp > 0.0 else 0.0
    ent_rec = tp / (tp + fn) if tp + fn > 0.0 else 0.0
    ent_f1 = 2 * ent_pre * ent_rec / (ent_pre + ent_rec) if ent_pre + ent_rec > 0.0 else 0.0
    print("Precision:\t{:.6f}\tRecall:\t{:.6f}\tF1:{:.6f}".format(ent_pre, ent_rec, ent_f1))

    return ent_f1


# Training Function
def do_train(model, optimizer, train_loader, dev_loader, eval_every, num_epochs, model_file_path,
             log_file_path, model_card_name, device, best_valid_f1=-1):
    # initialize running values
    running_loss = 0.0
    global_step = 0
    train_loss_list = []
    global_steps_list = []
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

    pbar = tqdm(total=len(train_loader) * num_epochs)
    # training loop
    model.train()
    for epoch in range(num_epochs):

        for step, batch in enumerate(train_loader):
            src_ids, src_mask, tgt_ids, labels, ent_ids, ent_texts = batch
            inputs = {
                'input_ids': src_ids.to(device),
                'attention_mask': src_mask.to(device),
                'labels': tgt_ids[:, 1:].to(device)
            }
            output = model(**inputs)
            loss = output.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1
            pbar.update(1)

            del src_ids, src_mask, tgt_ids, labels, ent_ids, ent_texts, batch, output

            # evaluation step
            if global_step % eval_every == 0 and epoch > (1 * num_epochs / 3):
                # if global_step % eval_every == 0:
                model.eval()
                truths = []
                predictions = []
                ent_ids = []
                ent_texts = []

                with torch.no_grad():

                    # validation loop
                    for v_src_ids, v_src_mask, v_tgt_ids, v_labels, v_ent_ids, v_ent_texts in dev_loader:

                        inputs = {
                            'input_ids': v_src_ids.to(device),
                            'attention_mask': v_src_mask.to(device),
                        }

                        batch_beams = model.generate(
                            inputs['input_ids'].to(device),
                            max_length=id_max_length,
                            prefix_allowed_tokens_fn=restrict_decode_vocab,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id
                        )
                        if batch_beams.shape[-1] < id_max_length:
                            batch_beams = _pad_tensors_to_max_len(batch_beams, id_max_length)

                        batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], -1)
                        output_tokens = batch_beams[:, 1:].tolist()
                        predictions.extend(output_tokens)

                        truths.extend(v_labels[:, 1:].cpu().tolist())
                        ent_ids.extend(v_ent_ids)
                        ent_texts.extend(v_ent_texts)
                        # print(generated_outputs)

                        # valid_running_loss += loss.item()

                v_f1 = compute_metrics(predictions, truths)

                # evaluation
                average_train_loss = running_loss / eval_every
                train_loss_list.append(average_train_loss)
                global_steps_list.append(global_step)
                model.train()

                # print progress
                print(
                    'Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid F1: {:.4f}'
                    .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                            average_train_loss, v_f1))

                # resetting running values
                running_loss = 0.0

                # checkpoint
                if best_valid_f1 < v_f1:
                    best_valid_f1 = v_f1
                    save_checkpoint(model_file_path + '/' + f"model_{model_card_name}.pt", model,
                                    best_valid_f1)

                    res = []
                    for p, l, ent_id, ent_text in zip(predictions, truths, ent_ids, ent_texts):
                        res.append([ent_id, ent_text, l, p, 1 if l == p else 0])
                    with open(log_file_path + f'/log_{model_card_name}_{epoch}.json', "w") as f_out:
                        json.dump(res, f_out)
                        f_out.close()

            elif global_step % eval_every == 0:
                # print progress
                average_train_loss = running_loss / eval_every
                print(
                    'Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                            average_train_loss))

                # resetting running values
                running_loss = 0.0

    pbar.close()
    print(f'Finished training on all training data. Best F1 is {round(best_valid_f1, 4)}.')


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


    def restrict_decode_vocab(batch_idx, prefix_beam):
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

    if "use_claim" in args and args.use_claim:
        args.model_card_name = args.model_card_name + "_w_claim"

    if args.training:
        if args.read_model:
            load_checkpoint(args.model_dir + '/' + f"model_{args.model_card_name}.pt", model, device=args.device)
            model.eval()
        print("Now the training is running...")

        # Load data

        train_data, max_src_len, max_tgt_len = load_passage_pairs(args.train_passage_path, tokenizer, cate="train",
                                                                  max_seq_length=1024)
        train_data = Dataset.to_list(train_data["train"])
        if args.use_name:
            co_train_data, _1, _2 = load_span_pairs(args.train_name_path, tokenizer, max_seq_length=1024)
            max_src_len = max(max_src_len, _1)
            max_tgt_len = max(max_tgt_len, _2)
            train_data = train_data + co_train_data

        if args.use_synonym:
            co_train_data, _1, _2 = load_span_pairs(args.train_synonym_path, tokenizer, max_seq_length=1024)
            max_src_len = max(max_src_len, _1)
            max_tgt_len = max(max_tgt_len, _2)
            train_data = train_data + co_train_data

        if "use_claim" in args and args.use_claim:
            co_train_data, _1, _2 = load_span_pairs(args.train_claim_path, tokenizer, max_seq_length=1024)
            max_src_len = max(max_src_len, _1)
            max_tgt_len = max(max_tgt_len, _2)
            train_data = train_data + co_train_data

        train_data = Dataset.from_list(train_data)

        dev_data, _1, _2 = load_passage_pairs(args.dev_passage_path, tokenizer, cate="dev",
                                              max_seq_length=1024)
        max_src_len = max(max_src_len, _1)
        max_tgt_len = max(max_tgt_len, _2)

        train_loader = DataLoader(train_data, batch_size=args.per_device_train_batch_size,
                                  shuffle=True, collate_fn=collate_fn, drop_last=False)

        dev_loader = DataLoader(dev_data["dev"], batch_size=args.per_device_eval_batch_size, shuffle=False,
                                collate_fn=collate_fn, drop_last=False)

        # Set model
        model.to(args.device)

        for name, param in model.named_parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        print('# of Parameters: %d' % (
            sum([param.numel() if param.requires_grad is True else 0 for param in model.parameters()])))
        do_train(model=model, optimizer=optimizer, train_loader=train_loader, dev_loader=dev_loader,
                 eval_every=len(train_loader), num_epochs=args.num_train_epochs, model_file_path=args.model_dir,
                 log_file_path=args.log_dir, model_card_name=args.model_card_name, device=args.device)
    print('Finished.')
