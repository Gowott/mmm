export CUDA_VISIBLE_DEVICES=1
name="inbatch_freezeclip_bge25e-6_2nd_cliphn"
checkpoint_path="./checkpoint/${name}"

python gen_embeddings.py \
--out_path "${checkpoint_path}" \
--checkpoint "${checkpoint_path}/model.best.pt" \
--img_feat_path "../data/imgs.tsv" \
--img_linelist_path "../data/imgs.lineidx.new" \
--doc_path "../data/all_docs.json" \
--cap_path "../data/all_imgs.json" \
--query_path "../data/test.json" \
--txt_pretrain bge \
--img_pretrain clip \
--num_workers 10 \
--encode_query \
--encode_img \
--encode_txt \
--freeze \

mv "${checkpoint_path}/query_embedding.pkl" "${checkpoint_path}/test_query_embedding.pkl"

