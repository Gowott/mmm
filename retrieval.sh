#!/bin/bash
name="inbatch_freezeclip_bge25e-6_2nd_cliphn"
checkpoint_path="./checkpoint/${name}"


python retrieval.py \
--query_embed_path "./checkpoint/${name}/test_query_embedding.pkl" \
--img_embed_path "./checkpoint/${name}/img_embedding.pkl" \
--qrel_path "../data/test_qrels.txt" \
--dim 768 \
--out_path "${checkpoint_path}"

python retrieval.py \
--query_embed_path "${checkpoint_path}/test_query_embedding.pkl" \
--doc_embed_path "${checkpoint_path}/txt_embedding.pkl" \
--img_embed_path "${checkpoint_path}/img_embedding.pkl" \
--qrel_path "../data/test_qrels.txt" \
--dim 768 \
--out_path "${checkpoint_path}"



python retrieval.py \
--query_embed_path "${checkpoint_path}/test_query_embedding.pkl" \
--doc_embed_path "${checkpoint_path}/txt_embedding.pkl" \
--qrel_path "../data/test_qrels.txt" \
--dim 768 \
--out_path "${checkpoint_path}"

