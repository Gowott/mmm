import json
from visual import TSVFile
import logging
import os
from tqdm import tqdm
import torch
import argparse
import pickle
import base64
from PIL import Image
import io
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from data import load_caps, load_docs, load_file, WebQADataset
from PIL import ImageFile
from tqdm import tqdm
import clip
from model.model import biencoder
from transformers import AutoTokenizer, RobertaModel, BertModel, BertTokenizer, \
    RobertaTokenizer, AutoModel, CLIPModel
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import Blip2Processor, Blip2Model



ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger()

class ImgDataset(Dataset):
    def __init__(self, args, captions=None, txt_pretrain='dpr', img_pretrain='clip'):
        self.max_seq_len = args.max_seq_len
        self.txt_pretrain = txt_pretrain
        self.img_pretrain = img_pretrain

        self.img_map = {}
        self.img_ids = []
        self.captions = captions
        if img_pretrain == 'clip':
            model, preprocess = clip.load("ViT-B/32", jit=False)  # Must set jit=False for training
            self.preprocess = preprocess
        elif img_pretrain == 'blip':
            # self.preprocess = Blip2Processor.from_pretrained("../pretrain_weights/blip2-flant5_xl")
            model, preprocess = clip.load("ViT-B/32", jit=False)  # Must set jit=False for training
            self.preprocess = preprocess

        if txt_pretrain == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('../pretrain_weights/bert')
        elif txt_pretrain == 'dpr':
            self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("../pretrain_weights/dpr/dpr_ctx")
            self.qry_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("../pretrain_weights/dpr/dpr_q")
        elif txt_pretrain == 'bge':
            self.tokenizer = AutoTokenizer.from_pretrained('../pretrain_weights/bge')

        all_img_num = 0
        with open(args.img_linelist_path) as fin:
            for line in fin:
                tokens = line.strip().split('\t')
                all_img_num += 1
                self.img_map[tokens[0]] = int(tokens[1])
                self.img_ids.append(tokens[0])
        self.img_tsv = TSVFile(args.img_feat_path, all_img_num)

    def __len__(self):
        return len(self.img_ids)

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
        img_inputs = []
        img_caps = []
        idx_list = []

        for example in batch:
            img_inputs.append(example['img_inputs'])
            if 'img_caps' in example:
                img_caps.append(example['img_caps'])
            idx_list.append(example['idx'])
        processed_batch = {}
        processed_batch['idx_list'] = idx_list

        if self.img_pretrain == 'clip':
            img_inputs = torch.stack(img_inputs)
        elif self.img_pretrain == 'blip':
            # img_inputs = img_inputs
            # img_inputs = torch.stack(img_inputs)
            # img_inputs = img_inputs.squeeze(1)
            img_inputs = torch.stack(img_inputs)


        processed_batch['img_inputs'] = img_inputs
        if len(img_caps) != 0:
            #truncation='longest_first',  # Truncate to max_length
            assert len(img_caps) == len(img_inputs)
            if self.txt_pretrain == 'dpr':
                processed_batch['img_caps'] = self.ctx_tokenizer(img_caps, padding='max_length', max_length=128,
                                                                 return_tensors='pt', truncation=True).data
            else:
                processed_batch['img_caps'] = self.tokenizer(img_caps, padding='max_length', max_length=128,
                                                             return_tensors='pt', truncation=True).data

        return processed_batch

    def __getitem__(self, index):
        img_idx = self.img_ids[index]
        img_inputs = self.encode_img(img_idx)
        instance = {
            'idx': img_idx,
            'img_inputs': img_inputs['img']
        }
        if 'cap' in img_inputs:
            instance['img_caps'] = img_inputs['cap']

        return instance

class TextDataset(Dataset):
    def __init__(self, data, max_len, type=None, txt_pretrain='dpr', prompt=None):
        self.max_len = max_len
        self.data = data
        self.type = type
        self.txt_pretrain = txt_pretrain
        self.prompt_flag = prompt

        if txt_pretrain == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('../pretrain_weights/bert')
        elif txt_pretrain == 'dpr':
            self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("../pretrain_weights/dpr/dpr_ctx")
            self.qry_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("../pretrain_weights/dpr/dpr_q")
        elif txt_pretrain == 'bge':
            self.tokenizer = AutoTokenizer.from_pretrained('../pretrain_weights/bge')


    def __len__(self):
        return len(self.data)


    def Collector(self, batch):
        txt_inputs = []
        idx_list = []

        for qid, example in enumerate(batch):
            txt_inputs.append(example['txt_inputs'])
            idx_list.append(example['idx'])

        if self.type == 'query':
            if self.txt_pretrain == 'dpr':
                txt_inputs = self.qry_tokenizer(txt_inputs, padding=True, truncation=True, max_length=72, return_tensors='pt').data  # max_length:70
            elif self.txt_pretrain == 'bge':
                if not self.prompt_flag:
                    instruction = 'Represent this sentence for searching relevant passages:'
                else:
                    instruction = ''
                txt_inputs = self.tokenizer([instruction + q for q in txt_inputs], padding=True, truncation=True, max_length=90, return_tensors='pt').data
            else:
                txt_inputs = self.tokenizer(txt_inputs, padding=True, truncation=True, max_length=72, return_tensors='pt').data #max_length:70
        elif self.type == 'doc':
            if self.txt_pretrain == 'dpr':
                txt_inputs = self.ctx_tokenizer(txt_inputs, padding='max_length', max_length=128, return_tensors='pt', truncation=True).data
            else:
                txt_inputs = self.tokenizer(txt_inputs, padding='max_length', max_length=128, return_tensors='pt', truncation=True).data
        else:
            raise ValueError("wrong input type")

        processed_batch = {
            'txt_inputs': txt_inputs,
            'idx_list': idx_list
        }
        return processed_batch

    def __getitem__(self, index):
        example = self.data[index]
        txt_inputs = example[1]

        return {
            'idx': example[0],
            'txt_inputs': txt_inputs
        }




def gen_embeddings(model, valid_reader, outpath, type=None, freeze=None):
    model.eval()
    all_embeddings = []
    all_index = []
    for step, batch in tqdm(enumerate(tqdm(valid_reader, desc='generate embedding', ncols=60))):

        with torch.no_grad():
            idx_list = batch['idx_list']

            if freeze:
                if (args.txt_pretrain == 'dpr' or args.txt_pretrain == 'bge') and model.freeze_vis:
                    if type == 'query':
                        embeddings = model.encode_query(batch['txt_inputs'], device)
                    elif type == 'doc':
                        embeddings = model.encode_text(batch['txt_inputs'], device)
                    elif type == 'img':
                        if 'img_inputs' in batch:
                            img_inputs = batch['img_inputs'].to(device)
                            cap_inputs = batch['img_caps']
                            embeddings = model(input_type='img', x=img_inputs, device=device, cap=cap_inputs)
                    else:
                        raise ValueError("wrong input type")
            else:
                if 'img_inputs' in batch:
                    img_embeddings = model.encode_image(batch['img_inputs'].to(device))
                    if 'img_caps' in batch:
                        cap_embeddings = model.encode_text(batch['img_caps'], device)
                        embeddings = img_embeddings + cap_embeddings
                if 'txt_inputs' in batch:
                    embeddings = model.encode_text(batch['txt_inputs'], device)

            embeddings = F.normalize(embeddings, dim=-1)
            embeddings = embeddings.cpu()
            assert len(embeddings) == len(idx_list)
            all_index.extend(idx_list)
            all_embeddings.append(embeddings)
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    with open(outpath, 'wb') as fout:
        pickle.dump((all_index, all_embeddings), fout)




def load_docs(path):
    data = []
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            did = str(example['snippet_id'])
            title = example['title']
            doc = title + ' [SEP] ' + example['fact']
            data.append([did, doc])
    return data


def load_queries(path):
    data = []
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            qid = str(example['qid'])
            data.append([qid, example['Q']])
    return data


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("")
    parser.add_argument("--max_seq_len", type=int, default=77)

    parser.add_argument("--out_path", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--img_feat_path", type=str)
    parser.add_argument("--img_linelist_path", type=str)

    parser.add_argument("--query_path", type=str)
    parser.add_argument("--doc_path", type=str)
    parser.add_argument("--cap_path", type=str)

    parser.add_argument('--encode_txt', action='store_true', default=False)
    parser.add_argument('--encode_img', action='store_true', default=False)
    parser.add_argument('--encode_query', action='store_true', default=False)

    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--txt_pretrain", type=str, default='dpr')
    parser.add_argument("--img_pretrain", type=str, default='clip')
    parser.add_argument("--freeze", action='store_true', default=False)
    parser.add_argument("--share_weights", action='store_true', default=False)

    parser.add_argument("--only_vis_prompter", action='store_true', default=False)
    parser.add_argument("--woprompt", action='store_true', default=False)
    parser.add_argument("--only_cap", action='store_true', default=False)



    args = parser.parse_args()


    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    handlers = [logging.FileHandler(os.path.join(args.out_path, 'train_log.txt')), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    if args.share_weights:
        share_weights = True
    else:
        share_weights = False
    model = biencoder(freeze_vis=args.freeze, txt_pretrain_model=args.txt_pretrain,
                      img_pretrain_model=args.img_pretrain, share_weights=share_weights, args=args)

    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.to(device)
    docs = load_docs(args.doc_path)



    if args.encode_query:
        output = os.path.join(args.out_path, 'query_embedding.pkl')
        if not os.path.isfile(output):
            queries = load_queries(args.query_path)
            query_data = TextDataset(queries, args.max_seq_len, type='query', txt_pretrain=args.txt_pretrain, prompt=args.woprompt)
            query_sampler = SequentialSampler(query_data)
            query_reader = DataLoader(dataset=query_data, sampler=query_sampler, num_workers=args.num_workers,
                                        batch_size=args.batch_size, collate_fn=query_data.Collector)


            gen_embeddings(model, query_reader, output, type='query', freeze=args.freeze)

    if args.encode_img:
        output = os.path.join(args.out_path, 'img_embedding.pkl')
        if not os.path.isfile(output):
            captions = None
            if args.cap_path:
                captions = load_caps(args.cap_path)
            img_data = ImgDataset(args, captions=captions, txt_pretrain=args.txt_pretrain, img_pretrain=args.img_pretrain)
            sampler = SequentialSampler(img_data)
            img_reader = DataLoader(dataset=img_data, sampler=sampler, num_workers=args.num_workers,
                                          batch_size=args.batch_size, collate_fn=img_data.Collector)
            output = os.path.join(args.out_path, 'img_embedding.pkl')

            gen_embeddings(model, img_reader, output, type='img', freeze=args.freeze)

    if args.encode_txt:
        output = os.path.join(args.out_path, 'txt_embedding.pkl')
        if not os.path.isfile(output):
            docs = load_docs(args.doc_path)
            txt_data = TextDataset(docs, args.max_seq_len, type='doc', txt_pretrain=args.txt_pretrain)
            txt_sampler = SequentialSampler(txt_data)
            txt_reader = DataLoader(dataset=txt_data, sampler=txt_sampler, num_workers=args.num_workers,
                                        batch_size=args.batch_size, collate_fn=txt_data.Collector)

            output = os.path.join(args.out_path, 'txt_embedding.pkl')

            gen_embeddings(model, txt_reader, output, type='doc', freeze=args.freeze)