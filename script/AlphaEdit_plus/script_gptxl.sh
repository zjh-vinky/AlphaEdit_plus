export CUDA_VISIBLE_DEVICES=2


python3 -m experiments.evaluate \
    --alg_name=AlphaEdit_plus \
    --model_name=../models/gpt2-xl \
    --hparams_fname=gpt2-xl.json \
    --ds_name=zsre \
    --dataset_size_limit=200 \
    --num_edits=100 \
    --downstream_eval_steps=1

