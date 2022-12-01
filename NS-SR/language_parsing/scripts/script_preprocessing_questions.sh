# Questions

# train set
#--output_vocab_path ./output_data/rewrite/STAR_human_train_vocab.json \
# python ./tools/preprocess_questions.py \
#     --question_path /Users/yushoubin/Desktop/ActionQA/SR_Dataset/question_answer/v0.98/train/STAR_train.json \
#     --output_h5_path ./output_data/STAR_human_train.h5 \
#     --split train \
#     --question_type mc_question \
  
# # # val set
# python ./tools/preprocess_questions.py \
#     --question_path /Users/yushoubin/Desktop/ActionQA/SR_Dataset/question_answer/v0.98/val/STAR_val.json \
#     --output_h5_path ./output_data/STAR_human_val.h5 \
#     --split val \
#     --question_type mc_question \
    
# # test set
python ./tools/preprocess_questions.py \
    --question_path /Users/yushoubin/Desktop/v0.98_rewrite/STAR_rewrite_test_0.3.json \
    --output_h5_path ./output_data/STAR_human_test_0.3.h5 \
    --split test \
    --question_type mc_question \

python ./tools/preprocess_questions.py \
    --question_path /Users/yushoubin/Desktop/v0.98_rewrite/STAR_rewrite_test_0.5.json \
    --output_h5_path ./output_data/STAR_human_test_0.5.h5 \
    --split test \
    --question_type mc_question \