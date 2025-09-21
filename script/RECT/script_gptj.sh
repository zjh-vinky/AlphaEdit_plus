export CUDA_VISIBLE_DEVICES=4

python3 -m experiments.evaluate \
    --alg_name=MEMIT_rect \
    --model_name=../models/gpt2-j \
    --hparams_fname=EleutherAI_gpt-j-6B.json \
    --ds_name=cf \
    --dataset_size_limit=200 \
    --num_edits=100 \
    --downstream_eval_steps=1


