# Multiple-choice choices

#export TEMPLATE_NAME=descriptive_T1
#GPU_ID=$0
#--gpu_ids ${GPU_ID} \

CUDA_VISIBLE_DEVICES=0 python ./tools/run_train_exp.py \
    --run_dir rs_exps \
    --num_workers 1 \
    --num_iter 50000 \
    --load_checkpoint_path ./rs_exps/checkpoint_best.pt \
    --display_every 100 \
    --checkpoint_every 2000 \
    --dataset sr_dataset \
    --mc_q_train_question_path ./output_data/rewrite/STAR_human_train.h5 \
    --mc_q_val_question_path ./output_data/rewrite/STAR_human_test.h5 \
    --mc_q_vocab_path ./output_data/STAR_human_train_vocab.json \
    --batch_size 64 \
    