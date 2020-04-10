set -o errexit 
set -o pipefail
# setup environment
# . set_environment.sh
# Argument handling
model_folder_name=$1
[ -z "$model_folder_name" ] && model_folder_name=""
flag=$2
if [ -z "$flag" ];then
    flag=""
elif [ "$flag" != "--copy-model" ];then
    echo -e "\nbash $0 [<model_name> [--copy-model]]\n"
    exit 1
fi
set -o nounset 

EXTERNAL_FOLDER=/dccstor/ykt-parse/SHARED/MODELS/AMR/transition-amr-parser/
LOCAL_FOLDER=DATA/AMR/

# If model not provided, list available (seed averages)
[ "$model_folder_name" == "" ] && \
    echo -e "\nModels available in $EXTERNAL_FOLDER" && \
    python scripts/stack-transformer/rank_model.py \
        --checkpoints $EXTERNAL_FOLDER/models && \
    exit 1
    # --seed-average && \

# name of folder where checlpoints are
model_folder=$EXTERNAL_FOLDER/models/$model_folder_name/

# Load config
[ ! -f "$model_folder/config.sh" ] && \
    echo "Expected $model_folder/config.sh" && \
    echo "Have you included e. g. -seed42 in the name?" && \
    exit 1
. "$model_folder/config.sh"

# Check of model exists in remote
source_oracle_folder="$EXTERNAL_FOLDER/oracles/${ORACLE_TAG}/"
source_features_folder="$EXTERNAL_FOLDER/features/${ORACLE_TAG}_${PREPRO_TAG}/"
source_checkpoints_dir_root="$EXTERNAL_FOLDER/models/$model_folder_name"

target_oracle_folder="$LOCAL_FOLDER/oracles/${ORACLE_TAG}/"
target_features_folder="$LOCAL_FOLDER/features/${ORACLE_TAG}_${PREPRO_TAG}/"
target_checkpoints_dir_root="$LOCAL_FOLDER/models/$model_folder_name"

# Make folders if needed
mkdir -p $LOCAL_FOLDER/oracles/
mkdir -p $LOCAL_FOLDER/features/
mkdir -p $LOCAL_FOLDER/models/

# ORACLE
if [ -d "$target_oracle_folder" ];then
    echo "Skiping existing: $target_oracle_folder"
else
    echo "ln -s $source_oracle_folder $LOCAL_FOLDER/oracles/"
    ln -s $source_oracle_folder $LOCAL_FOLDER/oracles/
fi

# FEATURES
if [ -d "$target_features_folder" ];then
    echo "Skiping existing: $target_features_folder"
else
    echo "ln -s $source_features_folder $LOCAL_FOLDER/features/"
    ln -s $source_features_folder $LOCAL_FOLDER/features/
fi

# MODEL
if [ -d "$target_checkpoints_dir_root" ];then
    echo "Skiping existing: $target_checkpoints_dir_root"
else
    if [ "$flag" == "--copy-model" ];then
        echo "cp -R $source_checkpoints_dir_root $LOCAL_FOLDER/models/"
        cp -R $source_checkpoints_dir_root $LOCAL_FOLDER/models/
    else
        echo "ln -s $source_checkpoints_dir_root $LOCAL_FOLDER/models/"
        ln -s $source_checkpoints_dir_root $LOCAL_FOLDER/models/
    fi
fi
