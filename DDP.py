import sys
import torch
import logging
import time
import json
import numpy as np
from torch.utils.data._utils.collate import default_convert
from torch.utils.data import RandomSampler, DataLoader, Dataset, SequentialSampler, TensorDataset, random_split
from tqdm import tqdm
import time
from colorama import Fore
import os
import random
from utils import load_count_dict, load_cycle, negative_relation_sampling, load_text, load_paths, load_triplets_with_pos_tails, \
    myConvert,reshape_relation_prediction_ranking_data,element_wise_cos,cal_metrics,CosineEmbeddingLoss,load_rules, merge_rules_and_paths, \
    synchronize_metrics, transpose, get_logger, init_seeds, load_pos_triplets, cal_metrics_modified, mask_pos
from torch.cuda.amp import GradScaler, autocast
from transformers import set_seed
from models import SentenceTransformer
import torch.nn.functional as F
import argparse
from torch.optim import lr_scheduler
from collections import defaultdict

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DSampler


parser = argparse.ArgumentParser(description='Relation Prediction')

parser.add_argument('--device', type=str, default='cuda:0',
                    help='CUDA device or CPU')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--eval_metric', type=int, default=1, metavar='N',
                    help='index of evaluation metric [MR,MRR,Hit@1,Hit@3,Hit@10]')
parser.add_argument('--dataset', type=str, default='FB15k-237-subset',
                    help='name of the dataset')
parser.add_argument('--path_dir', type=str, default=None,
                    help='location of extracted paths for each triplet')
parser.add_argument('--text_dir', type=str, default=None,
                    help='location of relation and entity texts')
parser.add_argument('--model_load_file', type=str, default=None,
                    help='location to load pretrained cycle model')
parser.add_argument('--model_save_dir', type=str, default=None,
                    help='location to save model')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--model', type=str, default='sentence-transformers/all-mpnet-base-v2',
                    help='sentence transformer model name on Hugging Face website (huggingface.co)')
parser.add_argument('--tokenizer', type=str, default='sentence-transformers/all-mpnet-base-v2',
                    help='tokenizer name on Hugging Face website (huggingface.co)')
parser.add_argument('--train_sample_num', type=int, default=-1,
                    help='number of training samples randomly sampled, use -1 for all data')
parser.add_argument('--valid_sample_num', type=int, default=-1,
                    help='number of validating samples randomly sampled, use -1 for all data')
parser.add_argument('--max_path_num', type=int, default=3,
                    help='number of paths loaded for each triplet')
parser.add_argument('--neg_sample_num_train', type=int, default=5,
                    help='number of negative training samples')
parser.add_argument('--neg_sample_num_valid', type=int, default=5,
                    help='number of negative validating samples')
parser.add_argument('--neg_sample_num_test', type=int, default=50,
                    help='number of negative testing samples')
parser.add_argument('--mode', type=str, default='tail',
                    help='whether head or tail is fixed')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--suffix', type=str, default="_full",
                    help='suffix of the train file name')
parser.add_argument('--no_train',action='store_true', default=False,
                    help='whether train or not')
parser.add_argument('--no_test',action='store_true', default=False,
                    help='whether test or not')
parser.add_argument('--output_dir', type=str, default=None,
                    help='location to output test results')
parser.add_argument('--text_file', type=str, default='GoogleWikipedia',
                    help='long text to be loaded')
parser.add_argument('--text_length', type=int, default=256,
                    help='long text length')
parser.add_argument('--no_merge', action='store_true', default=False,
                    help='long text length')

# parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--local-rank", default=-1, type=int)#this work with python -m torch.distributed.launch; "args.local_rank"
parser.add_argument("--merge_max_path_num", default=5, type=int, help="use when ours.")
parser.add_argument("--AP_load_setting", type=str, help="ours/origin/cp")
parser.add_argument("--pos_triplets_path", type=str)

args = parser.parse_args()




############# prepare data ##########
class train_Dataset(Dataset):
    def __init__(self, data):
        self.data = data 
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, index):
        triplets, paths, label = self.data[index] 
        # list of neg_sample_num str for cur q; 
        # list of neg_sample_num lists, each contains str(each str is a path for cur triplet); 
        # int of cur q.
        return {"triplets":triplets,
                "paths": paths,
                "label": label} # labels: bsz
    
class eval_Dataset(Dataset):
    def __init__(self, data):
        self.data = data 
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, index):
        triplets, paths, label, mask = self.data[index] 
        # list of neg_sample_num str for cur q; 
        # list of neg_sample_num lists, each contains str(each str is a path for cur triplet); 
        # int of cur q.
        # list of list of 0/1 -> 0 for true ans of cur q.
        return {"triplets":triplets,
                "paths": paths,
                "label": label,
                "mask": mask}


cur_time = time.strftime("%Y-%m-%d_%H-%M-%S")
cur_time += f"_{args.AP_load_setting}"
os.makedirs(f"save/{args.dataset}{args.suffix}/relation_prediction_{args.mode}/{cur_time}", exist_ok=True)
os.makedirs(f"output/{args.dataset}{args.suffix}/relation_prediction_{args.mode}/{cur_time}", exist_ok=True)

if args.path_dir is None:
    path_dir = os.path.join("data/relation_prediction_path_data/", args.dataset, f'ranking_{args.mode}{args.suffix}/')
else:
    path_dir=args.path_dir

if args.text_dir is None:
    text_dir = os.path.join("data/data", args.dataset)
else:
    text_dir=args.text_dir

if args.model_load_file is None:
    model_load_file = os.path.join(f"save/{args.dataset}{args.suffix}",f"relation_prediction_{args.mode}", cur_time, f"{args.AP_load_setting}_ep%s.pth")
else:
    model_load_file=args.model_load_file

if args.model_save_dir is None:
    model_save_dir=os.path.join(f"save/{args.dataset}{args.suffix}", f"relation_prediction_{args.mode}", cur_time)
else:
    model_save_dir=args.model_save_dir
if args.output_dir is None:
    output_dir=os.path.join(f"output/{args.dataset}{args.suffix}", f"relation_prediction_{args.mode}", cur_time)
else:
    output_dir=args.output_dir


logger = get_logger(model_save_dir)

#################### init DDP ##################
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
init_seeds(args.seed + local_rank) # set seed for each rank
logger.info(f"set seed={args.seed + local_rank} for rank{local_rank}")

if local_rank == 0:
    for arg,val in vars(args).items():
        logger.info(f"{arg}: {val}")
    logger.info(f"performing experiment for {args.AP_load_setting} method...")

all_pos_tails = defaultdict(lambda: defaultdict(list))
train_triplets = load_triplets_with_pos_tails(os.path.join(path_dir, "ranking_train.txt"),args.mode,all_pos_tails,args.neg_sample_num_train)
train_paths = load_paths(os.path.join(path_dir, "relation_paths_train.txt"),
                         os.path.join(path_dir, "entity_paths_train.txt"), len(train_triplets),args.max_path_num,args.mode)
raw_valid_triplets = load_triplets_with_pos_tails(os.path.join(path_dir, "ranking_valid.txt"),args.mode,all_pos_tails,args.neg_sample_num_valid)
valid_paths = load_paths(os.path.join(path_dir, "relation_paths_valid.txt"),
                         os.path.join(path_dir, "entity_paths_valid.txt"), len(raw_valid_triplets),args.max_path_num,args.mode)
raw_ranking_triplets = load_triplets_with_pos_tails(os.path.join(path_dir, "ranking_test.txt"),args.mode,all_pos_tails,args.neg_sample_num_test)
ranking_paths = load_paths(os.path.join(path_dir, "relation_paths_test.txt"),
                        os.path.join(path_dir, "entity_paths_test.txt"), len(raw_ranking_triplets),args.max_path_num,args.mode)

if args.AP_load_setting == "origin":
    train_rules = load_rules(os.path.join(path_dir, "relation_rules_train.txt"),
                            os.path.join(path_dir, "rules_heads_train.txt"), len(train_triplets),args.max_path_num,train_triplets,args.mode)
    valid_rules = load_rules(os.path.join(path_dir, "relation_rules_valid.txt"),
                            os.path.join(path_dir, "rules_heads_valid.txt"), len(raw_valid_triplets),args.max_path_num,raw_valid_triplets,args.mode)
    ranking_rules = load_rules(os.path.join(path_dir, "relation_rules_test.txt"),
                            os.path.join(path_dir, "rules_heads_test.txt"), len(raw_ranking_triplets),args.max_path_num,raw_ranking_triplets,args.mode)
    
    # logger.info("loaded rules from relation_rules and rules_heads txt.")
    
    AP_only_empty = True
    train_merged=merge_rules_and_paths(args.no_merge,train_paths,train_rules,args.neg_sample_num_train,args.max_path_num, AP_only_empty)
    valid_merged=merge_rules_and_paths(args.no_merge,valid_paths,valid_rules,args.neg_sample_num_valid,args.max_path_num, AP_only_empty)
    ranking_merged=merge_rules_and_paths(args.no_merge,ranking_paths,ranking_rules,args.neg_sample_num_test,args.max_path_num, AP_only_empty)

elif args.AP_load_setting == "ours":
    train_rules = json.load(open(os.path.join(path_dir, "merge_AP_train.json"), "r"))
    valid_rules = json.load(open(os.path.join(path_dir, "merge_AP_valid.json"), "r"))
    ranking_rules = json.load(open(os.path.join(path_dir, "merge_AP_test.json"), "r"))
    # logger.info("loaded rules from merge rules json.")
    AP_only_empty = False
    train_merged=merge_rules_and_paths(args.no_merge,train_paths,train_rules,args.neg_sample_num_train,args.merge_max_path_num, AP_only_empty)
    valid_merged=merge_rules_and_paths(args.no_merge,valid_paths,valid_rules,args.neg_sample_num_valid,args.merge_max_path_num, AP_only_empty)
    ranking_merged=merge_rules_and_paths(args.no_merge,ranking_paths,ranking_rules,args.neg_sample_num_test,args.merge_max_path_num, AP_only_empty)

elif args.AP_load_setting == "cp":
    train_rules = [[] for _ in range(len(train_paths))]
    valid_rules = [[] for _ in range(len(valid_paths))]
    ranking_rules = [[] for _ in range(len(ranking_paths))]
    # logger.info(f"set all rules to empty (only use CP, remove all AP)")
    AP_only_empty = True # either T or F is ok.
    train_merged=merge_rules_and_paths(args.no_merge,train_paths,train_rules,args.neg_sample_num_train,args.max_path_num, AP_only_empty)
    valid_merged=merge_rules_and_paths(args.no_merge,valid_paths,valid_rules,args.neg_sample_num_valid,args.max_path_num, AP_only_empty)
    ranking_merged=merge_rules_and_paths(args.no_merge,ranking_paths,ranking_rules,args.neg_sample_num_test,args.max_path_num, AP_only_empty)

# logger.info(f"merge paths and rules fin. add AP only when CP is empty: {AP_only_empty}")

text,relation_texts = load_text(text_dir,args.text_file,args.text_length)
all_dict = {**text['entity'], **text['relation']}

# random in reshape_relation_prediction_ranking_data
train_triplets,train_merged,train_labels,_=reshape_relation_prediction_ranking_data(train_triplets,train_merged,args.neg_sample_num_train,all_dict,text)
valid_triplets,valid_merged,valid_labels,_=reshape_relation_prediction_ranking_data(raw_valid_triplets,valid_merged,args.neg_sample_num_valid,all_dict,text)
ranking_triplets,ranking_merged,ranking_labels,ranking_indexes=reshape_relation_prediction_ranking_data(raw_ranking_triplets,ranking_merged,args.neg_sample_num_test,all_dict,text)


pos_triplets = load_pos_triplets(args.pos_triplets_path, args.mode) #easy edges in ind G: h -> r -> [t]

valid_mask = mask_pos(pos_triplets, valid_labels, raw_valid_triplets, args.mode, args.neg_sample_num_valid)
ranking_mask = mask_pos(pos_triplets, ranking_labels, raw_ranking_triplets, args.mode, args.neg_sample_num_test)


train_data = list(zip(train_triplets, train_merged,train_labels))
valid_data = list(zip(valid_triplets, valid_merged,valid_labels,valid_mask))
test_data = list(zip(ranking_triplets, ranking_merged,ranking_labels,ranking_mask))


# DDP originally init here. moved above.

train_dataset = train_Dataset(train_data)
train_sampler = DSampler(train_dataset)
train_data_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

valid_dataset = eval_Dataset(valid_data)
valid_sampler = DSampler(valid_dataset, shuffle=False) #add shuffle false
valid_data_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=1)

test_dataset = eval_Dataset(test_data)
test_sampler = DSampler(test_dataset, shuffle=False) #add shuffle false
test_data_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)

scaler = torch.amp.GradScaler()


raw_model = SentenceTransformer(tokenizer_name=args.tokenizer,model_name=args.model, device=local_rank)
model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = torch.optim.AdamW(lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9, params=optimizer_grouped_parameters)
scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5)

criterion = CosineEmbeddingLoss # this is a func. #loss func obj to local_rank

def train():
    for epoch in range(1, args.epochs + 1):
        # ============================================ TRAINING ============================================================
        if local_rank == 0:
            logger.info(f"start to train epoch {epoch}")
        train_data_loader.sampler.set_epoch(epoch)
        training_pbar = tqdm(total=args.train_sample_num if args.train_sample_num>0 else len(train_data),
                             position=0, leave=True,
                             file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.WHITE, Fore.RESET))
        model.train()
        tr_loss = 0
        nb_tr_steps = 0

        for step, batch in enumerate(train_data_loader):
            triplets, paths, targets = batch["triplets"], batch["paths"], batch["label"]
            triplets, paths, targets, _ = transpose((triplets, paths, targets, None))

            targets = torch.tensor(targets).to(local_rank)

            bsz = len(triplets) # triplets: bsz * cand_num
            cand_num = len(triplets[0])
            path_num = len(paths[0][0]) # paths: bsz * cand_num * path_num
            
            cat_triplets = []
            for tri in triplets:
                cat_triplets.extend(tri)
            cat_paths = []

            for path in paths:
                for p in path:
                    cat_paths.extend(p)
            
            # path_inputs = model.module.tokenize(cat_paths) # input_ids:on cuda, shape=[len(cat_paths),seq_len]
            # tri_inputs = model.module.tokenize(cat_triplets)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):#The constructor for class "autocast" is deprecated  `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead     
                # tri_embeds = model(tri_inputs) #[cand_num*bsz, 768]
                tri_embeds = model(cat_triplets)
                tri_embeds = tri_embeds.view(bsz, cand_num, tri_embeds.shape[-1]).unsqueeze(dim=-2) #[bsz, can_num, 768] -> [bsz, can_num, 1, 768]
                # path_embeds = model(path_inputs) #[can_num*path_num*bsz, 768]
                path_embeds = model(cat_paths)
                path_embeds = path_embeds.view(bsz, cand_num, path_num, path_embeds.shape[-1]) #[bsz, can_num, path_num, 768]

                sim = torch.cosine_similarity(tri_embeds, path_embeds, dim=-1) #sim.shape=[bsz, can_num, path_num]
                loss = criterion(sim, targets) # targets.shape=[bsz]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()
            nb_tr_steps += 1
            training_pbar.update(len(targets))
        training_pbar.close()
        scheduler.step()
        if local_rank == 0:
            logger.info(f"Learning rate={optimizer.param_groups[0]['lr']}\nTraining loss={tr_loss / nb_tr_steps:.4f}")
        
        if (epoch+1) % 1 == 0: #3
            validate(epoch)


def validate(epoch):
    global best_ep
    global best_val_acc

    valid_pbar = tqdm(total=args.valid_sample_num if args.valid_sample_num>0 else len(valid_data),
                     position=0, leave=True,
                     file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
    model.eval()
    metric_tensor = torch.zeros((6,), device=local_rank)
    for batch in valid_data_loader:
        triplets, paths, targets, masks = batch["triplets"], batch["paths"], batch["label"], batch["mask"]
        triplets, paths, targets, masks = transpose((triplets, paths, targets, masks))
        
        targets = torch.tensor(targets, device=local_rank)
        masks = torch.tensor(masks, dtype=torch.bool, device=local_rank)

        bsz = len(triplets) # triplets: bsz * cand_num
        cand_num = len(triplets[0])
        path_num = len(paths[0][0]) # paths: bsz * cand_num * path_num
            
        cat_triplets = []
        for tri in triplets:
            cat_triplets.extend(tri)

        cat_paths = []
        for path in paths:
            for p in path:
                cat_paths.extend(p)
        
        with torch.no_grad():
            tri_embeds = model(cat_triplets)
            tri_embeds = tri_embeds.view(bsz, cand_num, tri_embeds.shape[-1]).unsqueeze(dim=-2) #[bsz, can_num, 768] -> [bsz, can_num, 1, 768]
            path_embeds = model(cat_paths)
            path_embeds = path_embeds.view(bsz, cand_num, path_num, path_embeds.shape[-1]) #[bsz, can_num, path_num, 768]

        sim = torch.cosine_similarity(tri_embeds, path_embeds, dim=-1) #sim.shape=[bsz, can_num, path_num]
        scores, _ = torch.max(sim, dim=-1) #-> tensor[bsz, can_num], choose the best path for each cand. return values tensor & indices tensor.
        cur_metrics = torch.tensor(cal_metrics_modified(scores, targets, masks)+[1], device=local_rank)
        metric_tensor += cur_metrics
        valid_pbar.update(len(targets))
    valid_pbar.close()

    dist.barrier()
    dist.reduce(metric_tensor, 0) #reduce to rank 0
    
    save = False
    if local_rank == 0:
        metrics = {}
        metric_names = ["MR", "MRR", "Hits@1", "Hits@3", "Hits@10"]
        for i, name in enumerate(metric_names):
            metrics[name] = (metric_tensor[i]/metric_tensor[-1]).item()
        cur_val_acc = metrics[metric_names[args.eval_metric]]

        logger.info(f"val scores of epoch{epoch}: " + ", ".join([f"{key}={val:.4f}" for key,val in metrics.items()]))

        if cur_val_acc > best_val_acc:      
            logger.info("< valid data > find ckpt with better metrics {0}={1:.4f} in EP{2}, saving to {3}_ep{4}.pth.".format(
                metric_names[args.eval_metric], cur_val_acc, epoch, args.AP_load_setting, epoch
                ))
            best_val_acc = cur_val_acc
            best_ep = epoch
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            # torch.save(model.state_dict(), os.path.join(model_save_dir, f"best_val_{args.AP_load_setting}.pth"))          
            torch.save(model.module.state_dict(), os.path.join(model_save_dir, f"{args.AP_load_setting}_ep{epoch}.pth")) #do not save in ddp mode   
            save = True 
            logger.info(f"< test data > eval cur best ckpt on test set:")

    save = torch.tensor(int(save), device=local_rank)
    dist.broadcast(save, src=0)
    if save.item() == 1:      
        test()

def test():
    global best_ep
    global best_val_acc

    ranking_pbar = tqdm(total=len(ranking_triplets),
                        position=0, leave=True,
                        file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET))
    
    model.eval()
    metric_tensor = torch.zeros((6,), device=local_rank)
    for batch in test_data_loader:
        triplets, paths, targets, masks = batch["triplets"], batch["paths"], batch["label"], batch["mask"]
        triplets, paths, targets, masks = transpose((triplets, paths, targets, masks))
        
        targets = torch.tensor(targets, device=local_rank)
        masks = torch.tensor(masks, dtype=torch.bool, device=local_rank)

        bsz = len(triplets) # triplets: bsz * cand_num
        cand_num = len(triplets[0])
        path_num = len(paths[0][0]) # paths: bsz * cand_num * path_num
            
        cat_triplets = []
        for tri in triplets:
            cat_triplets.extend(tri)

        cat_paths = []
        for path in paths:
            for p in path:
                cat_paths.extend(p)
        
        with torch.no_grad():
            tri_embeds = model(cat_triplets)
            tri_embeds = tri_embeds.view(bsz, cand_num, tri_embeds.shape[-1]).unsqueeze(dim=-2) #[bsz, can_num, 768] -> [bsz, can_num, 1, 768]
            path_embeds = model(cat_paths)
            path_embeds = path_embeds.view(bsz, cand_num, path_num, path_embeds.shape[-1]) #[bsz, can_num, path_num, 768]

        sim = torch.cosine_similarity(tri_embeds, path_embeds, dim=-1) #sim.shape=[bsz, can_num, path_num]
        scores, _ = torch.max(sim, dim=-1) #-> tensor[bsz, can_num], choose the best path for each cand. return values tensor & indices tensor.
        cur_metrics = torch.tensor(cal_metrics_modified(scores, targets, masks)+[1], device=local_rank)
        metric_tensor += cur_metrics
        ranking_pbar.update(len(targets))
    ranking_pbar.close()

    dist.barrier()
    dist.reduce(metric_tensor, 0) #reduce to rank 0
    
    if local_rank == 0:
        metrics = {}
        metric_names = ["MR", "MRR", "Hits@1", "Hits@3", "Hits@10"]
        for i, name in enumerate(metric_names):
            metrics[name] = (metric_tensor[i]/metric_tensor[-1]).item()

        logger.info(f"test scores: " + ", ".join([f"{key}={val:.4f}" for key,val in metrics.items()]))



best_ep = -1
best_val_acc = 0

if local_rank == 0:
    logger.info("test with random init model for ablation study:")
test()

if args.no_train is False:
    if args.model_load_file and dist.get_rank() == 0:
        # if save as ddp mode
        # model.load_state_dict(torch.load(model_load_file))
        # else
        raw_model.load_state_dict(torch.load(model_load_file))
        logger.info(f"Before Train: loaded model from {model_load_file}")

    model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    train()

# if args.no_test is False:
#     if dist.get_rank() == 0:
#         # if save as ddp mode
#         # model.load_state_dict(torch.load(model_load_file))
#         # else
#         if args.model_load_file and best_ep == -1:
#             raw_model.load_state_dict(torch.load(model_load_file))
#             logger.info(f"Before Test: loaded model from {model_load_file}")
#         else:
#             raw_model.load_state_dict(torch.load(model_load_file % best_ep))
#             logger.info(f"Before Test: loaded model from {model_load_file % best_ep}")

#     model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
#     test()   


