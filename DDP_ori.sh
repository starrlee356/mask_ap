set -v
dataset="WN18RR-subset-inductive" #FB15k-237-subset-inductive, NELL-995-subset-inductive, WN18RR-subset-inductive
suffix="_full"
finding_mode="head" #"head"
device="cuda:0"
seed=42
epochs=30 #30
relation_prediction_lr=1e-5

path_search_depth=5
path_support_type=2
path_support_threshold=1e-4

text_file='GoogleWikipedia'
text_length=48

min_search_depth=1
max_search_depth=2

rule_recall_threshold=0.5
rule_accuracy_threshold=0.5


# 10 DDP
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 --master-port 12345 DDP.py \
 --device $device --epochs $epochs --batch_size 2 \
 --dataset $dataset --learning_rate $relation_prediction_lr --neg_sample_num_train 5 \
 --neg_sample_num_valid 50 --neg_sample_num_test 50 --mode $finding_mode --seed $seed \
 --suffix $suffix --text_file $text_file --text_length $text_length \
 --AP_load_setting origin\
 --pos_triplets_path "data/data/$dataset/inductive_graph.txt" \
