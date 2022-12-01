# Parse Multiple-choice question

CUDA_VISIBLE_DEVICES=0 python ./tools/parse_mc.py \
    --run_dir rs_exps \
    --save_result_path ./result_data/parse_pg.json \
    --load_checkpoint_path ./rs_exps/checkpoint_best.pt \
    --dataset sr_dataset \
    --split  val \
    --mc_q_test_question_path ./output_data/STAR_human_test.h5 \
    --mc_q_val_question_path ./output_data/STAR_human_test.h5 \
    --mc_q_vocab_path ./output_data/STAR_human_train_vocab.json \
    --batch_size  16 \
    