
mmf_predict config=projects/srqa/srtransformer/defaults.yaml model=sr_transformer dataset=srqa run_type=test checkpoint.resume_file=./save/models/model_5.ckpt training.batch_size=1 env.report_dir=./executor/SR_Dataset/situation_graph/

python ./executor/test_parser.py   