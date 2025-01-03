# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2024-12-27 15:01
#    @Description   : 
#
# ===============================================================


import argparse
parser = argparse.ArgumentParser()
# ========================= training==========================
parser.add_argument('--warmup_steps', type=int, default=1000)
parser.add_argument('--warmup_proportion', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_epochs', default=10, type=int)
parser.add_argument('--accumulate_grad_batches', default=1, type=int)
parser.add_argument('--seed', default=12, type=int)
parser.add_argument('--eval_delay', default=0, type=int)
parser.add_argument('--precision', default=32, type=int)
parser.add_argument('--plugins', type=str, default='ddp_sharded')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--kfold', type=int, default=1)
parser.add_argument('--recompute', action='store_true')
parser.add_argument('--ls_eps', default=0., type=float)

# ========================= Data ==========================
parser.add_argument('--train_file', type=str, required=False)
parser.add_argument('--dev_file', type=str, required=False)
parser.add_argument('--predict_file', type=str, required=False)
parser.add_argument('--noise_prob', default=0., type=float)
parser.add_argument('--max_source_length', default=512, type=int)
parser.add_argument('--max_target_length', default=300, type=int)
parser.add_argument('--beams', default=3, type=int)
parser.add_argument('--num_workers', type=int, default=4)

# ========================= Model ==========================
parser.add_argument('--model_path', type=str)
parser.add_argument('--model_type', type=str)
parser.add_argument('--rdrop', action='store_true')
parser.add_argument('--rdrop_coef', default=5, type=float)
parser.add_argument('--output_dir', type=str, default='./output')