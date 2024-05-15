#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

name="inbatch_freezeblip_bge25e-6_incontext_1st"
checkpoint_path="./checkpoint/${name}"

python gen_embeddings.py \
  --out_path "${checkpoint_path}" \
  --checkpoint "${checkpoint_path}/model.best.pt" \
  --img_feat_path "../data/imgs.tsv" \
  --img_linelist_path "../data/imgs.lineidx.new" \
  --doc_path "../data/all_docs.json" \
  --cap_path "../data/all_imgs.json" \
  --query_path "../data/train.json" \
  --encode_query \
  --txt_pretrain bge \
  --img_pretrain blip \
  --freeze \
  --num_workers 10 \
  --encode_img \
  --encode_txt \



mv "${checkpoint_path}/query_embedding.pkl" "${checkpoint_path}/train_query_embedding.pkl"

python gen_embeddings.py \
  --out_path "${checkpoint_path}" \
  --checkpoint "${checkpoint_path}/model.best.pt" \
  --img_feat_path "../data/imgs.tsv" \
  --img_linelist_path "../data/imgs.lineidx.new" \
  --doc_path "../data/all_docs.json" \
  --cap_path "../data/all_imgs.json" \
  --query_path "../data/dev.json" \
  --encode_query \
  --txt_pretrain bge \
  --img_pretrain blip \
  --freeze \
  --num_workers 10 \

mv "${checkpoint_path}/query_embedding.pkl" "${checkpoint_path}/dev_query_embedding.pkl"

python get_hard_negs_all.py \
  --query_embed_path "${checkpoint_path}/train_query_embedding.pkl" \
  --img_embed_path "${checkpoint_path}/img_embedding.pkl" \
  --txt_embed_path "${checkpoint_path}/txt_embedding.pkl" \
  --data_path "../data/train.json" \
  --out_path "${checkpoint_path}/train_all.json" \
  --dim 768

python get_hard_negs_all.py \
  --query_embed_path "${checkpoint_path}/dev_query_embedding.pkl" \
  --img_embed_path "${checkpoint_path}/img_embedding.pkl" \
  --txt_embed_path "${checkpoint_path}/txt_embedding.pkl" \
  --data_path "../data/dev.json" \
  --out_path "${checkpoint_path}/dev_all.json" \
  --dim 768



python gen_embeddings.py \
  --out_path "${checkpoint_path}" \
  --checkpoint "${checkpoint_path}/model.best.pt" \
  --img_feat_path "../data/imgs.tsv" \
  --img_linelist_path "../data/imgs.lineidx.new" \
  --doc_path "../data/all_docs.json" \
  --cap_path "../data/all_imgs.json" \
  --query_path "../data/test.json" \
  --freeze \
  --txt_pretrain bge \
  --img_pretrain blip \
  --num_workers 10 \
  --encode_query \

mv "${checkpoint_path}/query_embedding.pkl" "${checkpoint_path}/test_query_embedding.pkl"