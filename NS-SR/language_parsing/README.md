# Question Parser
This repository provides a sample to train a Seq2Seq program pareser.

### Setup Enviroment
use a new conda env for parser because of different pytorch version.
`conda_env.yml`
### Download data
sh ./scripts/script_preprocessing_questions.sh
### Train a parser
sh ./scripts/train_sr_reasoning.sh
### Parse results
sh ./scripts/test_sr_reasoning.sh
