# Set variables and environment for a give experiment
#
# Variables intended to be use outside of this script are CAPITALIZED
#
set -o errexit
set -o pipefail
set -o nounset

# NOTE: Assumes you ran training with
#
# scripts/stack-transformer/configs/qald-prepro_o5+Word100_roberta.large.top24_stnp6x6.sh
#
# to get (see use below)
#
# DATA/AMR/features/ldcqbqaldlarge_o5+Word100_RoBERTa-large-top24/dict.en.txt
# DATA/AMR/features/ldcqbqaldlarge_o5+Word100_RoBERTa-large-top24/dict.actions.txt
# DATA/AMR/models/ldcqbqaldlarge_o5+Word100_RoBERTa-large-top24_stnp6x6-seed44/checkpoint_best_SMATCH.pt
#

TASK_TAG=AMR

# Global paths
AMR_CORPORA=$data_root/AMR/

# All data stored here
data_root=DATA/$TASK_TAG/

# AMR ORACLE
# See transition_amr_parser/data_oracle.py:argument_parser
AMR_TRAIN_FILE=$AMR_CORPORA/QB20200305/qb.pseudo.aln
AMR_DEV_FILE=$AMR_CORPORA/QB20200305/qald_dev2_pass3.jaln
AMR_TEST_FILE=$AMR_CORPORA/QB20200305/blindtest.jkaln
# WIKI files
# NOTE: If left empty no wiki will be added
WIKI_DEV=""
AMR_DEV_FILE_WIKI="" 
WIKI_TEST=""
AMR_TEST_FILE_WIKI=""
# Leave empty to create entity rules from the corpus
ENTITY_RULES=""

# Labeled shift: each time we shift, we also predict the word being shited
# but restrict this to top MAX_WORDS. Controlled by
# --multitask-max-words --out-multitask-words --in-multitask-words
# To have an action calling external lemmatizer (SpaCy)
# --copy-lemma-action
MAX_WORDS=100
ORACLE_TAG=qbqaldlargefinetune_o5+Word${MAX_WORDS}
ORACLE_FOLDER=$data_root/oracles/${ORACLE_TAG}/
ORACLE_TRAIN_ARGS="
    --multitask-max-words $MAX_WORDS 
    --out-multitask-words $ORACLE_FOLDER/train.multitask_words 
    --copy-lemma-action
"
ORACLE_DEV_ARGS="
    --in-multitask-words $ORACLE_FOLDER/train.multitask_words \
    --copy-lemma-action
"

# PREPROCESSING
# See fairseq/fairseq/options.py:add_preprocess_args
PREPRO_TAG="RoBERTa-large-top24"
# CCC configuration in scripts/stack-transformer/jbsub_experiment.sh
PREPRO_GPU_TYPE=v100
PREPRO_QUEUE=x86_24h
FEATURES_FOLDER=$data_root/features/${ORACLE_TAG}_${PREPRO_TAG}/
# TODO: Get this paths refred to the SHARED folder
# ${AMR_MODELS}/features/qaldlarge_extracted/
srcdict="$data_root/features/ldcqbqaldlarge_o5+Word100_RoBERTa-large-top24/dict.en.txt"
tgtdict="$data_root/features/ldcqbqaldlarge_o5+Word100_RoBERTa-large-top24/dict.actions.txt"
FAIRSEQ_PREPROCESS_ARGS="
    --source-lang en
    --target-lang actions
    --trainpref $ORACLE_FOLDER/train
    --validpref $ORACLE_FOLDER/dev
    --testpref $ORACLE_FOLDER/test
    --destdir $FEATURES_FOLDER
    --workers 1 
    --pretrained-embed roberta.large
    --bert-layers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    --srcdict $srcdict
    --tgtdict $tgtdict
    --machine-type AMR 
    --machine-rules $ORACLE_FOLDER/train.rules.json 
    --fp16
"

# TRAINING
# See fairseq/fairseq/options.py:add_optimization_args,add_checkpoint_args
# model types defined in ./fairseq/fairseq/models/transformer.py
TRAIN_TAG=stnp6x6
base_model=stack_transformer_6x6_nopos
# number of random seeds trained at once
NUM_SEEDS=3
# CCC configuration in scripts/stack-transformer/jbsub_experiment.sh
TRAIN_GPU_TYPE=v100
TRAIN_QUEUE=ppc_24h
# --lazy-load for very large corpora (data does not fit into RAM)
# --bert-backprop do backprop though BERT
# NOTE: --save-dir is specified inside dcc/train.sh to account for the seed
MAX_EPOCH=190
CHECKPOINTS_DIR_ROOT="$data_root/models/${ORACLE_TAG}_${PREPRO_TAG}_${TRAIN_TAG}"
# NOTE: We start from a pretrained model
pretrained="$data_root/models/ldcqbqaldlarge_o5+Word100_RoBERTa-large-top24_stnp6x6-seed44/checkpoint_best_SMATCH.pt"
FAIRSEQ_TRAIN_ARGS="
    $FEATURES_FOLDER
    --restore-file $pretrained
    --max-epoch $MAX_EPOCH
    --arch $base_model
    --optimizer adam
    --adam-betas '(0.9,0.98)'
    --clip-norm 0.0
    --lr-scheduler inverse_sqrt
    --warmup-init-lr 1e-07
    --warmup-updates 4000
    --pretrained-embed-dim 1024
    --lr 0.0005
    --min-lr 1e-09
    --dropout 0.3
    --weight-decay 0.0
    --criterion label_smoothed_cross_entropy
    --label-smoothing 0.01
    --keep-last-epochs 100
    --max-tokens 3584
    --log-format json
    --reset-optimizer
"

# TESTING
# See fairseq/fairseq/options.py:add_optimization_args,add_checkpoint_args
# --path flag specified in the dcc/test.sh script
# --results-path is dirname from --path plus $TEST_TAG
beam_size=1
TEST_TAG="beam${beam_size}"
CHECKPOINT=checkpoint_best.pt
# CCC configuration in scripts/stack-transformer/jbsub_experiment.sh
TEST_GPU_TYPE=v100
TEST_QUEUE=x86_6h
FAIRSEQ_GENERATE_ARGS="
    $FEATURES_FOLDER 
    --gen-subset valid
    --machine-type AMR 
    --machine-rules $ORACLE_FOLDER/train.rules.json
    --beam ${beam_size}
    --batch-size 128
    --remove-bpe
"
# TODO: It would be cleaner to use the checkpoint path for --machine-rules but
# this can be externally provided on dcc/test.sh
