#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

name="inbatch_freezeclip_bge25e-6_1st"
checkpoint_path="./checkpoint/${name}"
pretrained_model_path="./checkpoint/${name}/model.best.pt"

output_name="inbatch_freezeclip_bge25e-6_2nd_cliphn"
output_checkpoint="./checkpoint/${output_name}"

accelerate launch --config_file accelerator_config.yaml ./train.py \
--train_path ../CLIP-DPR/checkpoint_multi_inb/train_all.json \
--valid_path ../CLIP-DPR/checkpoint_multi_inb/dev_all.json \
--doc_path "../data/all_docs.json" \
--cap_path "../data/all_imgs.json" \
--img_feat_path "../data/imgs.tsv" \
--img_linelist_path "../data/imgs.lineidx.new" \
--device 0 \
--train_batch_size 64 \
--valid_batch_size 256 \
--out_path "${output_checkpoint}" \
--learning_rate 25e-6 \
--linear_lr 25e-6 \
--num_train_epochs 30 \
--txt_pretrain bge \
--img_pretrain clip \
--early_stop 10 \
--num_workers 10 \
--freeze \
--img_neg_num 1 \
--txt_neg_num 1 \
--pretrained_model_path "${pretrained_model_path}" \

#--amb_hn \



#    parser.add_argument("--only_vis_prompter", action='store_true', default=False)
#    parser.add_argument("--woprompt", action='store_true', default=False)
#    parser.add_argument("--only_cap", action='store_true', default=False)







#--train_path "${checkpoint_path}/train_all.json" \
#--valid_path "${checkpoint_path}/dev_all.json" \

