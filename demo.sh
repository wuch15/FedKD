#!/bin/bash
export JAVA_HOME=/usr/jdk/jdk1.8.0_121 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/lib/server
export HADOOP_HOME=/usr/local/hadoop
export PATH=${PATH}:${HADOOP_HOME}/bin:${JAVA_HOME}/bin
export LIBRARY_PATH=${LIBRARY_PATH}:${HADOOP_HOME}/lib/native
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${JAVA_HOME}/jre/lib/amd64/server:${HADOOP_HOME}/lib/native:/usr/local/cuda-10.0/extras/CUPTI/lib64
source $HADOOP_HOME/libexec/hadoop-config.sh
export PYTHONPATH="/opt/conda/lib/python3.6/site-packages"
export CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob)

hvd_size=4
mode=$1 # ['train', 'test', 'train_test']

root_data_dir=../
train_date="train_valid"
test_date="test"
market="MIND"


epoch=2
num_attention_heads=20
news_attributes=$2


model_dir=$3
embedding_source="glove"
glove_embedding_path="glove.840B.300d.txt"
batch_size=8

user_log_mask=False
padded_news_different_word_index=False
use_padded_news_embedding=False

save_steps=10000
lr=0.0001
max_steps_per_epoch=120000
filter_num=1
mask_uet_bing_rate=0.8
npratio=4

# process UET
process_uet=$4
process_bing=False


# pretrain
use_pretrain_news_encoder=$5
pretrain_news_encoder_path='../model_all/pretrain_textencoder'

# debias
debias=$6
title_share_encoder=$7
uet_agg_method=$8

# turing
apply_turing=$9
word_embedding_dim=${10}

if [ ${mode} == train ] 
then
    mpirun -np ${hvd_size} -H localhost:${hvd_size} \
    python run.py --root_data_dir ${root_data_dir} \
    --mode ${mode} --epoch ${epoch} --market ${market} \
    --model_dir ${model_dir} --embedding_source ${embedding_source} \
    --glove_embedding_path ${glove_embedding_path} --batch_size ${batch_size} \
    --news_attributes ${news_attributes} --lr ${lr} \
    --padded_news_different_word_index ${padded_news_different_word_index} \
    --user_log_mask ${user_log_mask} --use_padded_news_embedding ${use_padded_news_embedding} \
    --train_date ${train_date} --test_date ${test_date} --save_steps ${save_steps} \
    --filter_num ${filter_num} --max_steps_per_epoch ${max_steps_per_epoch} \
    --process_uet ${process_uet} --process_bing ${process_bing} --mask_uet_bing_rate ${mask_uet_bing_rate} \
    --title_share_encoder ${title_share_encoder} \
    --use_pretrain_news_encoder ${use_pretrain_news_encoder} \
    --pretrain_news_encoder_path ${pretrain_news_encoder_path} \
    --uet_agg_method ${uet_agg_method} --debias ${debias} --word_embedding_dim ${word_embedding_dim} \
    --npratio ${npratio} --apply_turing ${apply_turing} --num_attention_heads ${num_attention_heads}
elif [ ${mode} == test ]
then
    batch_size=32
    log_steps=10
    load_ckpt_name=${11}
    CUDA_LAUNCH_BLOCKING=1 python run.py --root_data_dir ${root_data_dir} \
    --mode ${mode} --epoch ${epoch} --market ${market} \
    --model_dir ${model_dir} --embedding_source ${embedding_source} \
    --glove_embedding_path ${glove_embedding_path} --batch_size ${batch_size} \
    --news_attributes ${news_attributes} --lr ${lr} \
    --padded_news_different_word_index ${padded_news_different_word_index} \
    --user_log_mask ${user_log_mask} --use_padded_news_embedding ${use_padded_news_embedding} \
    --train_date ${train_date} --test_date ${test_date} --save_steps ${save_steps} \
    --process_uet ${process_uet} --process_bing ${process_bing} \
    --title_share_encoder ${title_share_encoder} \
    --use_pretrain_news_encoder ${use_pretrain_news_encoder} \
    --pretrain_news_encoder_path ${pretrain_news_encoder_path} \
    --uet_agg_method ${uet_agg_method} --debias ${debias} --word_embedding_dim ${word_embedding_dim} \
    --log_steps ${log_steps} --apply_turing ${apply_turing} --num_attention_heads ${num_attention_heads} \
    --load_ckpt_name ${load_ckpt_name}
fi