import json
import os
from visual import TSVFile
import torch
import random
import base64
from PIL import Image
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import torch
import clip
from PIL import ImageFile
from transformers import AutoTokenizer, RobertaModel, BertModel, BertTokenizer, \
    RobertaTokenizer, AutoModel, CLIPModel
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import Blip2Processor, Blip2Model


ImageFile.LOAD_TRUNCATED_IMAGES = True

class WebQADataset(Dataset):
    def __init__(self, args, data, docs, captions, shuffle, img_pretrain=None, txt_pretrain=None):
        self.img_neg_num = args.img_neg_num
        self.txt_neg_num = args.txt_neg_num
        self.shuffle = shuffle
        self.img_pretrain = img_pretrain

        if img_pretrain == 'clip':
            model, preprocess = clip.load("ViT-B/32", jit=False)  # Must set jit=False for training  ViT-B/32 ViT-B-32.pt
            self.preprocess = preprocess
        elif img_pretrain == 'blip':
            # self.preprocess = Blip2Processor.from_pretrained("../pretrain_weights/blip2-flant5_xl")
            model, preprocess = clip.load("ViT-B/32", jit=False)  # Must set jit=False for training  ViT-B/32 ViT-B-32.pt
            self.preprocess = preprocess

        if txt_pretrain == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('../pretrain_weights/bert')
        elif txt_pretrain == 'dpr':
            # self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("/home/student2020/dwz/UniVL-DR/pretrain_weights/dpr/dpr_ctx")
            # self.qry_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("/home/student2020/dwz/UniVL-DR/pretrain_weights/dpr/dpr_q")
            #
            self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("../pretrain_weights/dpr/dpr_ctx")
            self.qry_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("../pretrain_weights/dpr/dpr_q")

        elif txt_pretrain == 'bge':
            self.tokenizer = AutoTokenizer.from_pretrained('../pretrain_weights/bge')

        self.txt_pretrain = txt_pretrain
        self.img_map = {}
        self.img_tsv = []
        self.docs = docs
        self.captions = captions
        self.prompt_flag = args.woprompt

        img_feat_path = args.img_feat_path
        img_linelist_path = args.img_linelist_path
        all_img_num = 0
        with open(img_linelist_path) as fin:
            for line in fin:
                tokens = line.strip().split('\t')
                all_img_num += 1
                self.img_map[tokens[0]] = int(tokens[1])
        self.img_tsv = TSVFile(img_feat_path, all_img_num)
        self.data = data

    def __len__(self):
        return len(self.data)


    def encode_img(self, idx):
        offset = self.img_map[idx]
        img = self.img_tsv[offset][1]
        img = Image.open(io.BytesIO(base64.b64decode(img))).convert('RGB')
        if self.img_pretrain == 'clip':
            img = self.preprocess(img)
        elif self.img_pretrain == 'blip':
            # img = self.preprocess(images=img, return_tensors="pt")['pixel_values']
            img = self.preprocess(img)

        if self.captions != None:
            cap = self.captions[idx]
            return {'img': img, 'cap':cap}
        return {'img': img}


    def Collector(self, batch):
        queries = []
        img_inputs = []
        txt_inputs = []
        cap_inputs = []
        txt_labels = []
        img_labels = []
        processed_batch = {}
        for qid, example in enumerate(batch):
            queries.append(example['query'])
            if 'pos_img' in example:
                img_inputs.append(example['pos_img']['img'])
                if 'cap' in example['pos_img']:
                    cap_inputs.append(example['pos_img']['cap'])
                img_labels.append(qid)
            if 'pos_txt' in example:
                txt_inputs.append(example['pos_txt'])
                txt_labels.append(qid)
            if 'neg_imgs' in example:
                for instance in example['neg_imgs']:
                    img_inputs.append(instance['img'])
                    if 'cap' in instance:
                        cap_inputs.append(instance['cap'])
                    img_labels.append(-1)
            if 'neg_txts' in example:
                for instance in example['neg_txts']:
                    txt_inputs.append(instance)
                    txt_labels.append(-1)

        if self.txt_pretrain == 'dpr':
            processed_batch['queries'] = self.qry_tokenizer(queries, padding=True, truncation=True, max_length=72, return_tensors='pt').data  # max_length:70
        elif self.txt_pretrain == 'bge':
            if not self.prompt_flag:
                instruction = 'Represent this sentence for searching relevant passages:'
            else:
                instruction = ''
            processed_batch['queries'] = self.tokenizer([instruction + q for q in queries], padding=True, truncation=True, max_length=90, return_tensors='pt').data #max_length:70
        else:
            processed_batch['queries'] = self.tokenizer(queries, padding=True, truncation=True, max_length=72, return_tensors='pt').data  # max_length:70
        assert len(txt_inputs) != 0 or len(img_inputs) != 0
        if len(img_inputs) != 0:
            if self.img_pretrain == 'clip':
                img_inputs = torch.stack(img_inputs)
            elif self.img_pretrain == 'blip':
                # img_inputs = img_inputs
                # img_inputs = torch.stack(img_inputs)
                # img_inputs = img_inputs.squeeze(1)
                img_inputs = torch.stack(img_inputs)

            processed_batch['img_inputs'] = img_inputs
            processed_batch['img_labels'] = img_labels
            if len(cap_inputs) != 0:
                assert len(cap_inputs) == len(img_inputs)
                if self.txt_pretrain == 'dpr':
                    processed_batch['img_caps'] = self.ctx_tokenizer(cap_inputs, padding='max_length', max_length=128, return_tensors='pt', truncation=True).data
                else:
                    processed_batch['img_caps'] = self.tokenizer(cap_inputs, padding='max_length', max_length=128, return_tensors='pt', truncation=True).data

        if len(txt_inputs) != 0:
            if self.txt_pretrain == 'dpr':
                processed_batch['txt_inputs'] = self.ctx_tokenizer(txt_inputs, padding='max_length', max_length=128, return_tensors='pt', truncation=True).data
            else:
                processed_batch['txt_inputs'] = self.tokenizer(txt_inputs, padding='max_length', max_length=128, return_tensors='pt', truncation=True).data
            processed_batch['txt_labels'] = txt_labels

        return processed_batch

    def __getitem__(self, index):
        example = self.data[index]
        query = example['Q']
        instance = {'query': query}

        if len(example['img_posFacts']) != 0:
            if self.shuffle:
                idx = random.choice(example['img_posFacts'])
            else:
                idx = example['img_posFacts'][0]
            instance["pos_img"] = self.encode_img(idx)
        elif len(example['txt_posFacts']) != 0:
            if self.shuffle:
                idx = random.choice(example['txt_posFacts'])
            else:
                idx = example['txt_posFacts'][0]
            instance["pos_txt"] = self.docs[idx]
        else:
            raise ('No positive instance!')



        if self.img_neg_num > 0:
            neg_imgs = []
            neg_img_idx = example['img_negFacts']
            if self.shuffle:
                np.random.shuffle(neg_img_idx)
            neg_img_idx = neg_img_idx[:self.img_neg_num]
            for idx in neg_img_idx:
                neg_imgs.append(self.encode_img(idx))
            instance["neg_imgs"] = neg_imgs

        if self.txt_neg_num > 0:
            neg_txts = []
            neg_txt_idx = example['txt_negFacts']
            if self.shuffle:
                np.random.shuffle(neg_txt_idx)
            neg_txt_idx = neg_txt_idx[:self.txt_neg_num]
            for idx in neg_txt_idx:
                neg_txts.append(self.docs[idx])
            instance["neg_txts"] = neg_txts
        return instance




def load_file(path, txt=True, img=True):
    data = []
    with open(path) as fin:
        assert (txt or img)
        for line in fin:
            example = json.loads(line.strip())
            txt_negFacts = example['txt_negFacts']
            np.random.shuffle(txt_negFacts)
            example['txt_negFacts'] = txt_negFacts

            img_negFacts = example['img_negFacts']
            np.random.shuffle(img_negFacts)
            example['img_negFacts'] = img_negFacts

            if txt and len(example['txt_posFacts']) != 0:
                data.append(example)
            if img and len(example['img_posFacts']) != 0:
                data.append(example)
    return data

def load_docs(path):
    data = {}
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            did = str(example['snippet_id'])
            title = example['title']
            data[did] = title + ' [SEP] ' + example['fact']
            # data[did] = example['fact']
    return data

def load_caps(path):
    data = {}
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            imgid = str(example['image_id'])
            title = example['title']
            data[imgid] = title + ' [SEP] ' + example['caption']
    return data




def load_file_hn(path, txt=True, img=True):
    data = []
    with open(path) as fin:
        assert (txt or img)
        for line in fin:
            example = json.loads(line.strip())
            txt_negFacts = example['txt_negFacts']
            np.random.shuffle(txt_negFacts)
            example['txt_negFacts'] = txt_negFacts

            img_negFacts = example['img_negFacts']
            np.random.shuffle(img_negFacts)
            example['img_negFacts'] = img_negFacts

            if 'all_negFacts' in example:
                all_negFacts = example['all_negFacts']
                np.random.shuffle(all_negFacts)
                example['all_negFacts'] = all_negFacts

            if txt and len(example['txt_posFacts']) != 0:
                data.append(example)
            if img and len(example['img_posFacts']) != 0:
                data.append(example)
    return data

