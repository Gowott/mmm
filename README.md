# 3m


训练代码：sh ./train_multi.sh
生成难负例: sh ./get_hn.sh
检索推理: sh./retrieval.sh
生成embedding: sh ./gen_embed.sh

## 一阶段训练
train_multi.sh  中删除txt_num, img_num，更改train_data, valid_data的路径为/data/dwz/data
下的train.json和valid.json

## self-minded negatives
一阶段训练完使用该权重 运行sh ./get_hn.sh 采集难负例
采集完成后，修改img_num和txt_num以及train_data valid_data的路径再次执行 train_multi.sh

## 推理
推理使用sh ./retrieval.sh (推理前需确认query, txt, img的embedding是否已经提取，未提取使用gen_embed.sh提取）
