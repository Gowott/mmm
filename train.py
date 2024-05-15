import json
from visual import TSVFile
import logging
import os
import numpy as np
from tqdm import tqdm
import torch
import argparse
from torch import optim
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data import load_caps, load_docs, load_file, WebQADataset, load_file_hn
from model.model import biencoder
logger = logging.getLogger()
import random
import torch.nn.functional as F
from data import WebQADataset
from accelerate import Accelerator


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def eval_loss(model, loss_function, valid_reader):
    model.eval()
    total_loss = 0.0
    total_corr = 0.0
    counter = 0.0
    for step, batch in enumerate(tqdm(valid_reader, desc='eval processing', ncols=60, leave=False)):
        with torch.no_grad():
            query_embedding = model.encode_query(batch['queries'], device)
            candidate_embeddings = []
            all_labels = []
            pos_labels = [-1] * query_embedding.size(0)
            if (args.txt_pretrain == 'dpr' or args.txt_pretrain == 'bge') and model.freeze_vis:


                if 'img_inputs' in batch:
                    img_inputs = batch['img_inputs']
                    cap_inputs = batch['img_caps']
                    img_embeddings = model(input_type='img', x=img_inputs, device=device, cap=cap_inputs)

                    candidate_embeddings.append(img_embeddings)
                    all_labels.extend(batch['img_labels'])

                    if args.amb_hn:
                        tmp_all_labels = torch.tensor(all_labels)
                        indices = (tmp_all_labels != -1).nonzero().squeeze()
                        img_inputs = img_inputs[indices]
                        cap_inputs['input_ids'] = cap_inputs['input_ids'][indices]
                        cap_inputs['token_type_ids'] = cap_inputs['token_type_ids'][indices]
                        cap_inputs['attention_mask'] = cap_inputs['attention_mask'][indices]
                        amb_len = indices.size(0)

                        # fix cap, amb img
                        amb_img = torch.cat((img_inputs[1:], img_inputs[0].unsqueeze(0)), dim=0)
                        amb_img_emb = model(input_type='img', x=amb_img, device=device, cap=cap_inputs)
                        candidate_embeddings.append(amb_img_emb)
                        amb_labels = [-1] * amb_len
                        all_labels.extend(amb_labels)

                        # # #fix img, amb cap
                        # amb_cap = {}
                        # amb_cap['input_ids'] = torch.cat(
                        #     (cap_inputs['input_ids'][1:], cap_inputs['input_ids'][0].unsqueeze(0)), dim=0)
                        # amb_cap['token_type_ids'] = torch.cat(
                        #     (cap_inputs['token_type_ids'][1:], cap_inputs['token_type_ids'][0].unsqueeze(0)), dim=0)
                        # amb_cap['attention_mask'] = torch.cat(
                        #     (cap_inputs['attention_mask'][1:], cap_inputs['attention_mask'][0].unsqueeze(0)), dim=0)
                        # amb_cap_emb = model(input_type='img', x=img_inputs, device=device, cap=amb_cap)
                        # candidate_embeddings.append(amb_cap_emb)
                        # amb_labels = [-1] * amb_len
                        # all_labels.extend(amb_labels)



                if 'txt_inputs' in batch:
                    txt_embeddings = model.encode_text(batch['txt_inputs'], device)
                    candidate_embeddings.append(txt_embeddings)
                    all_labels.extend(batch['txt_labels'])



            else:
                if 'img_inputs' in batch:
                    img_embeddings = model.encode_image(batch['img_inputs'])
                    if 'img_caps' in batch:
                        cap_embeddings = model.encode_text(batch['img_caps'], device)
                        img_embeddings = img_embeddings + cap_embeddings
                    candidate_embeddings.append(img_embeddings)
                    all_labels.extend(batch['img_labels'])
                if 'txt_inputs' in batch:
                    txt_embeddings = model.encode_text(batch['txt_inputs'], device)
                    candidate_embeddings.append(txt_embeddings)
                    all_labels.extend(batch['txt_labels'])

            candidate_embeddings = torch.cat(candidate_embeddings, dim=0)
            for step, idx in enumerate(all_labels):
                if idx != -1:
                    pos_labels[idx] = step
            query_embedding = F.normalize(query_embedding, dim=-1)
            candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
            logit_scale = model.logit_scale.exp()
            score = torch.matmul(query_embedding, candidate_embeddings.t()) * logit_scale
            target = torch.tensor(pos_labels, dtype=torch.long).to(device)
            loss = loss_function(score, target)
            max_score, max_idxs = torch.max(score, 1)
            correct_predictions_count = (max_idxs == target).sum()/ query_embedding.size(0)
            total_corr += correct_predictions_count.item()
            total_loss += loss.item()
            counter += 1
    if counter == 0:
        return 0.0, 0.0
    return total_loss / counter, total_corr / counter

def train(train_reader, valid_reader, model, accelerator=None):
    t_total = len(train_reader) // args.gradient_accumulation_steps * args.num_train_epochs



    # #bias weight decay = 0.2
    # exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    # include = lambda n, p: not exclude(n, p)
    #
    # named_parameters = list(model.named_parameters())
    # gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    # rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    #
    # optimizer = optim.AdamW(
    #     [
    #         {"params": gain_or_bias_params, "weight_decay": 0.},
    #         {"params": rest_params, "weight_decay": 0.2},
    #     ],
    #     lr=args.learning_rate,
    #     betas=(0.9,  0.98),
    #     eps=1.0e-6,
    # )
    # scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup_steps, t_total)


    prompter_params = list(map(id, model.img_pooler.parameters()))  # 返回的是parameters的 内存地址
    txt_pooler = list(map(id, model.txt_pooler.parameters()))  # 返回的是parameters的 内存地址
    pretrain_params = filter(lambda p: id(p) not in prompter_params and id(p) not in txt_pooler, model.parameters())

    optimizer = optim.AdamW(
        [
            {"params": pretrain_params, 'lr': args.learning_rate},
            {"params": model.img_pooler.parameters(), 'lr': args.linear_lr},
            {"params": model.txt_pooler.parameters(), 'lr': args.linear_lr},
        ]
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, t_total)
    '''
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.

    '''


    loss_function = torch.nn.CrossEntropyLoss()
    tag, global_step, global_loss, best_acc = 0, 0, 0.0, 0.0
    model, optimizer, train_reader, scheduler = accelerator.prepare(model, optimizer, train_reader, scheduler)
    valid_reader = accelerator.prepare(valid_reader)
    model.zero_grad()
    if accelerator:
        device = accelerator.device

    epoch_cnt = 1
    with accelerator.autocast():
        for epoch in tqdm(range(int(args.num_train_epochs)), ncols=60):
            epoch_cnt += 1
            for step, batch in enumerate(tqdm(train_reader, leave=False, ncols=60)):
                model.train()
                query_embedding = model.encode_query(batch['queries'], device)
                candidate_embeddings = []
                all_labels = []
                pos_labels = [-1] * query_embedding.size(0)
                if (args.txt_pretrain == 'dpr' or args.txt_pretrain == 'bge') and model.freeze_vis:
                    # amb_flag = False
                    if 'img_inputs' in batch:
                        img_inputs = batch['img_inputs']
                        cap_inputs = batch['img_caps']
                        img_embeddings = model(input_type='img', x=img_inputs, device=device, cap=cap_inputs)
                        candidate_embeddings.append(img_embeddings)
                        all_labels.extend(batch['img_labels'])

                        del img_embeddings
                        if args.amb_hn:
                            tmp_all_labels = torch.tensor(all_labels)
                            indices = (tmp_all_labels != -1).nonzero().squeeze()
                            img_inputs = img_inputs[indices]
                            cap_inputs['input_ids'] = cap_inputs['input_ids'][indices]
                            cap_inputs['token_type_ids'] = cap_inputs['token_type_ids'][indices]
                            cap_inputs['attention_mask'] = cap_inputs['attention_mask'][indices]
                            amb_len = indices.size(0)


                            # fix cap, amb img
                            amb_img = torch.cat((img_inputs[1:], img_inputs[0].unsqueeze(0)), dim = 0)
                            amb_img_emb = model(input_type='img', x=amb_img, device=device, cap=cap_inputs)
                            candidate_embeddings.append(amb_img_emb)
                            amb_labels = [-1] * amb_len
                            all_labels.extend(amb_labels)

                            # # #fix img, amb cap
                            # amb_cap = {}
                            # amb_cap['input_ids'] = torch.cat((cap_inputs['input_ids'][1:], cap_inputs['input_ids'][0].unsqueeze(0)), dim=0)
                            # amb_cap['token_type_ids'] = torch.cat((cap_inputs['token_type_ids'][1:], cap_inputs['token_type_ids'][0].unsqueeze(0)), dim=0)
                            # amb_cap['attention_mask'] = torch.cat((cap_inputs['attention_mask'][1:], cap_inputs['attention_mask'][0].unsqueeze(0)), dim=0)
                            # amb_cap_emb = model(input_type='img', x=img_inputs, device=device, cap=amb_cap)
                            # candidate_embeddings.append(amb_cap_emb)
                            # amb_labels = [-1] * amb_len
                            # all_labels.extend(amb_labels)


                    if 'txt_inputs' in batch:
                        txt_embeddings = model.encode_text(batch['txt_inputs'], device)
                        candidate_embeddings.append(txt_embeddings)
                        all_labels.extend(batch['txt_labels'])

                        del txt_embeddings
                else:
                    if 'img_inputs' in batch:
                        img_embeddings = model.encode_image(batch['img_inputs'])
                        if 'img_caps' in batch:
                            cap_embeddings = model.encode_text(batch['img_caps'], device)
                            img_embeddings = img_embeddings + cap_embeddings
                        candidate_embeddings.append(img_embeddings)
                        all_labels.extend(batch['img_labels'])
                    if 'txt_inputs' in batch:
                        txt_embeddings = model.encode_text(batch['txt_inputs'], device)
                        candidate_embeddings.append(txt_embeddings)
                        all_labels.extend(batch['txt_labels'])

                candidate_embeddings = torch.cat(candidate_embeddings, dim=0)
                for step, idx in enumerate(all_labels):
                    if idx != -1:
                        pos_labels[idx] = step
                '''
                l1 = infoNCE(q, pos, img_hn+txt_hn+inbatch)
                l2 = infoNCE(q, pos_img, amb_img + amb_cap)
                '''


                query_embedding = F.normalize(query_embedding, dim=-1)
                candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
                logit_scale = model.logit_scale.exp()
                score = torch.matmul(query_embedding, candidate_embeddings.t()) * logit_scale
                target = torch.tensor(pos_labels, dtype=torch.long).to(device)
                loss = loss_function(score, target)
                max_score, max_idxs = torch.max(score, 1)
                correct_predictions_count = (max_idxs == target).sum() / query_embedding.size(0)
                accelerator.backward(loss)

                global_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    global_step += 1
                    # scheduler(global_step)
                    optimizer.step()
                    model.zero_grad()
                    # #更新学习率
                    scheduler.step()

                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, linear_lr: {:.6f}, acc: {:.4f} ({:.4f}), ".format(
                        epoch, global_step,
                        optimizer.param_groups[0]["lr"],
                        optimizer.param_groups[1]["lr"],
                        correct_predictions_count,
                        global_loss / global_step,
                    ))
                    # logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, acc: {:.4f} ({:.4f}), ".format(
                    #     epoch, global_step,
                    #     optimizer.param_groups[0]["lr"], correct_predictions_count,
                    #     global_loss / global_step,
                    # ))


                    # print('*************', global_loss, '****************')
                    if global_step % args.eval_steps == 0 and global_step > 0:
                        logger.info('*********Start eval loss**********')
                        dev_loss, dev_acc = eval_loss(model, loss_function, valid_reader)
                        logger.info("Evaluation at global step {}, average dev loss: {:.4f}, average dev acc: {:.4f}".format(
                            global_step, dev_loss, dev_acc))
                        if best_acc <= dev_acc:
                            best_acc = dev_acc
                            torch.save({'epoch': epoch,
                                        'model': model.state_dict()}, os.path.join(args.out_path, "model.best.pt"))
                            logger.info("Saved best epoch {0}, best acc {1}".format(epoch, best_acc))
                            tag = 0
                        else:
                            tag += 1
                        if tag >= args.early_stop:
                            logger.info('*********early stop**********')
                            return


if __name__ == '__main__':

    parser = argparse.ArgumentParser("")

    parser.add_argument("--out_path", type=str, default='./checkpoint/')
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--doc_path", type=str)
    parser.add_argument("--cap_path", type=str)
    parser.add_argument("--img_feat_path", type=str)
    parser.add_argument("--img_linelist_path", type=str)

    parser.add_argument('--only_txt', action='store_true', default=False)
    parser.add_argument('--only_img', action='store_true', default=False)


    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--valid_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--linear_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--img_neg_num", type=int, default=0)
    parser.add_argument("--txt_neg_num", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--txt_pretrain", type=str, default='dpr')
    parser.add_argument("--img_pretrain", type=str, default='clip')
    parser.add_argument("--freeze", action='store_true', default=False)
    parser.add_argument("--only_vis_prompter", action='store_true', default=False)
    parser.add_argument("--woprompt", action='store_true', default=False)
    parser.add_argument("--only_cap", action='store_true', default=False)
    parser.add_argument("--amb_hn", action='store_true', default=False)


    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device))

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    handlers = [logging.FileHandler(os.path.join(args.out_path, 'train_log.txt')), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logger.info(args)
    set_seed(args)

    # if args.only_txt:
    #     train_data = load_file(args.train_path, img=False)
    #     valid_data = load_file(args.valid_path, img=False)
    # elif args.only_img:
    #     train_data = load_file(args.train_path, txt=False)
    #     valid_data = load_file(args.valid_path, txt=False)

    second_stage = True
    if second_stage:
        train_data = load_file_hn(args.train_path)
        valid_data = load_file_hn(args.valid_path)
    else:
        train_data = load_file(args.train_path)
        valid_data = load_file(args.valid_path)
    docs = load_docs(args.doc_path)
    captions = None
    if args.cap_path:
        captions = load_caps(args.cap_path)


    model = biencoder(freeze_vis=args.freeze, txt_pretrain_model=args.txt_pretrain, img_pretrain_model=args.img_pretrain, args=args)

    train_data = WebQADataset(args, train_data, docs, captions=captions, shuffle=True,
                                  img_pretrain=args.img_pretrain, txt_pretrain=args.txt_pretrain)
    train_sampler = RandomSampler(train_data)
    traindata_reader = DataLoader(dataset=train_data, sampler=train_sampler, num_workers=args.num_workers,
                                  batch_size=args.train_batch_size, collate_fn=train_data.Collector, drop_last=True, pin_memory=True)

    valid_data = WebQADataset(args, valid_data, docs, captions=captions, shuffle=False,
                                  img_pretrain=args.img_pretrain, txt_pretrain=args.txt_pretrain)
    valid_sampler = SequentialSampler(valid_data)
    validdata_reader = DataLoader(dataset=valid_data, sampler=valid_sampler, num_workers=args.num_workers,
                                batch_size=args.valid_batch_size, collate_fn=valid_data.Collector, drop_last=False)

    if args.pretrained_model_path != None:
        logger.info('loading checkpoint from {}'.format(args.pretrained_model_path))
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location='cpu')['model'])

    accelerator = Accelerator(mixed_precision='no')
    device = accelerator.device
    model.to(device)

    train(traindata_reader, validdata_reader, model, accelerator)
