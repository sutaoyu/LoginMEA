gpu_id='0'
# 'FB15K_DB15K' 'FB15K_YAGO15K' 'zh_en' 'ja_en' 'fr_en'
dataset='FB15K_DB15K'
ratio=0.2
seed=2023
il_start=50
bsize=3500
adapter_choice=0
dropout=0.0
rank=8
use_GphForwards=1
other_modal_type=0
Is_LMFSoftmax=0
joint_type=1
mr_fusion_type=7
final_fusion_types=(0)
if [[ "$dataset" == *"FB"* ]]; then
    dataset_dir='mmkb-datasets'
    tau=0.1
    other_modal_type=0
else
    dataset_dir='DBP15K'
    tau=0.1
    ratio=0.3
fi
echo "Running with dataset=${dataset}, ratio=${ratio}"
current_datetime=$(date +"%Y-%m-%d-%H-%M")
head_name=${current_datetime}_${dataset}
file_name=${head_name}_${ratio}
echo ${file_name}
CUDA_VISIBLE_DEVICES=${gpu_id} python3 -u src/run.py \
    --file_dir data/${dataset_dir}/${dataset} \
    --pred_name ${file_name} \
    --rate ${ratio} \
    --lr .0005 \
    --epochs 1000 \
    --dropout ${dropout} \
    --hidden_units "300,300,300" \
    --check_point 50  \
    --bsize ${bsize} \
    --il \
    --il_start ${il_start} \
    --semi_learn_step 5 \
    --csls \
    --csls_k 3 \
    --seed ${seed} \
    --tau ${tau} \
    --tau2 4.0 \
    --structure_encoder "Dualmodal-joint-Mr" \
    --joint_type ${joint_type} \
    --img_dim 300 \
    --attr_dim 300 \
    --name_dim 100 \
    --char_dim 100 \
    --bi_adapter \
    --adapter_choice ${adapter_choice} \
    --use_ms_loss \
    --use_joint_loss \
    --joint_use_nce \
    --LMFrank ${rank} \
    --use_GphForward ${use_GphForward} \
    --add_other_modal ${other_modal_type} \
    --Is_LMFSoftmax ${Is_LMFSoftmax} \
    --mr_fusion_type ${mr_fusion_type} \
    --final_fusion_type ${final_fusion_type} \
    --use_proxy \
    --w_name \
    --w_char  > logs/${file_name}.log
echo $0 >> logs/${file_name}.log
