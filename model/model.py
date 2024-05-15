import torch
import torch.nn as nn
import clip
from transformers import AutoTokenizer, RobertaModel, BertModel, BertTokenizer, RobertaTokenizer, AutoModel, CLIPModel
from transformers import CLIPVisionModel, CLIPTextModelWithProjection
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers.modeling_utils import get_parameter_dtype
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2Model, Blip2QFormerModel, Blip2VisionModel
import torch



class biencoder(nn.Module):
    def __init__(self, txt_pretrain_model='bert', img_pretrain_model='clip', freeze_vis=False, share_weights=False, args=None):
        super(biencoder, self).__init__()
        self.args = args

        self.txt_pretrain = txt_pretrain_model
        self.img_pretrain = img_pretrain_model
        self.freeze_vis = freeze_vis
        self.share_weights = share_weights

        img_dim = 512
        if txt_pretrain_model == 'bert':
            self.txt_encoder = BertModel.from_pretrained('../pretrain_weights/bert')
            self.txt_encoder.pooler.dense = nn.Linear(768, 512)

        elif txt_pretrain_model == 'dpr':
            self.qry_encoder = DPRQuestionEncoder.from_pretrained('../pretrain_weights/dpr/dpr_q')
            self.txt_encoder = DPRContextEncoder.from_pretrained("../pretrain_weights/dpr/dpr_ctx")
            img_dim = 768
        elif txt_pretrain_model == 't5':
            pass
        elif txt_pretrain_model == 'flan_t5':
            pass
        elif txt_pretrain_model == 'bge':

            # tokenizer = AutoTokenizer.from_pretrained('/home/student2020/dwz/UniVL-DR/pretrain_weights/bge')
            if share_weights:
                self.txt_encoder = AutoModel.from_pretrained('../pretrain_weights/bge')
            else:
                self.qry_encoder = AutoModel.from_pretrained('../pretrain_weights/bge')
                self.txt_encoder = AutoModel.from_pretrained('../pretrain_weights/bge')
            img_dim = 768



        elif txt_pretrain_model == 'vilt':
            raise ValueError ("to be written")

        if img_pretrain_model == 'clip':
            '''
            clip-vit-patch16: /home/student2020/dwz/UniVL-DR/pretrain_weights/clip-vit-patch_16
            clip-vit-patch32: /home/student2020/dwz/UniVL-DR/pretrain_weights/clip
            '''
            self.img_token = 49 #[bs, 50, 512]
            self.preprocess = AutoProcessor.from_pretrained("../pretrain_weights/clip")
            model = CLIPModel.from_pretrained("../pretrain_weights/clip")
            self.img_encoder = model.vision_model

            if freeze_vis:
                for p in self.img_encoder.parameters():
                    p.requires_grad = False


            self.logit_scale = model.logit_scale
        elif img_pretrain_model == 'vilt':
            #vilt的embedding --> 输入dpr
            raise ValueError("to be written")
        elif img_pretrain_model == 'blip':
            self.preprocess = Blip2Processor.from_pretrained("../pretrain_weights/blip2-flant5_xl")
            model = Blip2Model.from_pretrained("../pretrain_weights/blip2-flant5_xl")
            del model.language_model
            del model.language_projection
            self.img_encoder = model
            self.img_token = 32
            if freeze_vis:
                for p in self.img_encoder.parameters(): #freeze q-former and clip
                    p.requires_grad = False
            model = CLIPModel.from_pretrained("../pretrain_weights/clip")
            self.logit_scale = model.logit_scale



        self.img_pooler = nn.Sequential(
            nn.Linear(768, img_dim),
            nn.ReLU(),
            nn.Linear(img_dim, img_dim)
        )

        self.txt_pooler = nn.Sequential(
            nn.Linear(768, 768)
        )

    def forward(self, input_type, x, device=None, cap=None):
        if input_type == 'query':
            return self.encode_query(x, device)
        elif input_type == 'txt':
            return self.encode_text(x, device)
        elif input_type == 'img':
            if self.freeze_vis == False:
                return self.encode_image(x)
            else:
                img_feat = self.get_img_feature(x)
                cap_feat, cap = self.get_cap_feature(cap, device)
                output = self.mmdpr(img_feat, cap_feat, cap)

                return output

        else:
            raise ValueError("input_type is not assigned")


    def encode_query(self, x, device):
        input_ids = x['input_ids'].to(device)
        token_type_ids = x['token_type_ids'].to(device)
        attention_mask = x['attention_mask'].to(device)
        if self.txt_pretrain == 'bert':
            output = self.txt_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output
        elif self.txt_pretrain == 'dpr' or self.txt_pretrain == 'bge':
            if self.share_weights:
                output = self.txt_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output
            else:
                output = self.qry_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output
        return output

    def encode_text(self, x, device):
        input_ids = x['input_ids'].to(device)
        token_type_ids = x['token_type_ids'].to(device)
        attention_mask = x['attention_mask'].to(device)
        output = self.txt_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output
        return output

    def encode_image(self, x):
        output = self.img_encoder(x).pooler_output
        output = self.img_pooler(output)
        return output

    #use dpr with frozen clip for mm retrieval
    def get_img_feature(self, x):
        if self.img_pretrain == 'clip':
            img_feat = self.img_encoder(x).last_hidden_state
            img_feat = img_feat[:, 1:, :] #[bs, 49, dim]
        elif self.img_pretrain == 'blip':
            img_feat = self.img_encoder.get_qformer_features(x).last_hidden_state


        output = self.img_pooler(img_feat)
        return output

    #bert embedding
    def get_cap_feature(self, x, device):
        input_ids = x['input_ids'].to(device)
        token_type_ids = x['token_type_ids'].to(device)
        attention_mask = x['attention_mask'].to(device)

        if self.txt_pretrain == 'dpr':
            output = self.txt_encoder.ctx_encoder.bert_model.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        elif self.txt_pretrain == 'bge':
            output = self.txt_encoder.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        return output, x

    def mmdpr(self, img_feat, cap_feat, cap):
        '''
        input: [cls] + img + cap
        padding: [cls] + cap -- > [cls] + img + cap
        mask: [cls] + cap -- > [cls] + img + cap
        '''
        device = img_feat.device
        if self.args.only_vis_prompter:
            token_type_ids = cap['token_type_ids'].to(device)
            attention_mask = cap['attention_mask'].to(device)
            # expand padding and mask matrix
            if len(img_feat.size()) == 2:
                img_feat = img_feat.unsqueeze(1)

            input_embedding = img_feat
            bs = img_feat.size(0)
            attention_mask = torch.ones(bs, self.img_token).to(img_feat.device)
            attention_mask = self.get_extended_attention_mask(attention_mask)
        elif self.args.only_cap:
            token_type_ids = cap['token_type_ids'].to(device)
            attention_mask = cap['attention_mask'].to(device)
            input_embedding = cap_feat
            attention_mask = self.get_extended_attention_mask(attention_mask)
        else:
            token_type_ids = cap['token_type_ids'].to(device)
            attention_mask = cap['attention_mask'].to(device)
            #expand padding and mask matrix
            if len(img_feat.size()) == 2:
                img_feat = img_feat.unsqueeze(1)

            input_embedding = torch.cat((cap_feat, img_feat), dim=1)
            attention_mask = self.expand_mask_matrix(attention_mask)
            attention_mask = self.get_extended_attention_mask(attention_mask)


        if self.txt_pretrain == 'dpr':
            outputs = self.txt_encoder.ctx_encoder.bert_model.encoder(input_embedding, attention_mask)
        elif self.txt_pretrain == 'bge':
            outputs = self.txt_encoder.encoder(input_embedding, attention_mask)


        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        output = self.txt_pooler(pooled_output)
        return output


    def expand_mask_matrix(self, attention_mask):
        bs = attention_mask.size(0)
        extend_mask = torch.ones(bs, self.img_token).to(attention_mask.device)
        output = torch.cat((attention_mask, extend_mask), dim=1)
        return output


    def get_extended_attention_mask(self, attention_mask, device = None, dtype = None):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]

        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    #get type
    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)




#
# if __name__=='__main__':
#     model = biencoder(txt_pretrain_model='dpr', img_pretrain_model='blip', freeze_vis=True, share_weights=False)
#     print()
