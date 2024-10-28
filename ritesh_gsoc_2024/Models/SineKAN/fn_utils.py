from config import TransformerConfig
from model import build_kanformer
from tokenizer import Tokenizer
from prefix_tokenizer import PrefixTokenizer
import torch.distributed as dist
import torch
import torch.nn as nn
import random
from typing import List
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import argparse
from datetime import timedelta


# Special tokens & coressponding ids
BOS_IDX, PAD_IDX, EOS_IDX, UNK_IDX, SEP_IDX = 0, 1, 2, 3, 4
special_symbols = ['<S>', '<PAD>', '</S>', '<UNK>', '<SEP>']


def create_tokenizer(df, config, index_pool_size, momentum_pool_size, is_old=False):
    if config.is_prefix:
        tokenizer = PrefixTokenizer(df,special_symbols, UNK_IDX)
    else:
        tokenizer = Tokenizer(df, index_pool_size, momentum_pool_size, special_symbols, UNK_IDX,config.to_replace, is_old=is_old)
    src_vocab = tokenizer.build_src_vocab(config.seed)
    src_itos = {value: key for key, value in src_vocab.get_stoi().items()}
    tgt_vocab = tokenizer.build_tgt_vocab()
    tgt_itos = {value: key for key, value in tgt_vocab.get_stoi().items()}

    return tokenizer, src_vocab, tgt_vocab, src_itos, tgt_itos


def init_distributed_mode(config):
    dist.init_process_group(backend=config.backend, timeout=timedelta(minutes=30))




def generate_unique_random_integers(x, start=0, end=3000):
    if x > (end - start + 1):
        raise ValueError(
            "x cannot be greater than the range of unique values available")
    return random.sample(range(start, end), x)


def tgt_decode(tgt: List[int], tgt_itos):
    out = ''
    for y in tgt:
        if y != PAD_IDX and y != BOS_IDX and y != EOS_IDX:
            out += tgt_itos[y]
    return out


def src_decode(src: List[int], src_itos):
    out = ''
    for y in src:
        if y != PAD_IDX and y != BOS_IDX and y != EOS_IDX:
            out += src_itos[y]
    return out

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def calculate_line_params(point1, point2):

    x1, y1 = point1
    x2, y2 = point2

    # Check if the x coordinates are the same to avoid division by zero
    if x1 == x2:
        raise ValueError(
            "The x coordinates of the two points must be different to define a straight line.")

    # Calculate the slope (m)
    m = (y2 - y1) / (x2 - x1)

    # Calculate the intercept (b)
    b = y1 - m * x1

    return m, b


def get_model(config):
    """
    Function to instantiate a Model object and initialize its parameters using 
    previously defined global variables.

    Returns:
        Model: Initialized model object.
    """
    model = build_kanformer(config.src_voc_size, config.tgt_voc_size,config.src_max_len,config.tgt_max_len, 
                            config.embedding_size, config.num_layers, 
                            config.nhead,config.dropout,config.d_ff,config.ff_dims,config.device)

    return model


def parse_ff_dims(ff_dims_str):
    return list(map(int, ff_dims_str.split(',')))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transformer training configuration")

    parser.add_argument('--project_name', type=str,
                        required=True, help='Project name')
    parser.add_argument('--run_name', type=str, required=True, help='Run name')
    parser.add_argument('--model_name', type=str,
                        required=True, help='Model name')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Root directory for  checkpoints')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory for data')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for training (e.g., "cuda" for GPU, "cpu" for CPU)')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Total number of epochs for training')
    parser.add_argument('--training_batch_size', type=int,
                        required=True, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int,
                        required=True, help='Batch size for testing')
    parser.add_argument('--valid_batch_size', type=int,
                        required=True, help='Batch size for validation')
    parser.add_argument('--num_workers', type=int, required=True,
                        help='Number of worker processes for data loading')
    parser.add_argument('--embedding_size', type=int,
                        required=True, help='Dimensionality of word embeddings')
    parser.add_argument('--nhead', type=int, required=True,
                        help='Number of attention heads in the transformer model')
    parser.add_argument('--num_layers', type=int, required=True,
                        help='Number of encoder & decoder layers in the transformer model')
    parser.add_argument('--d_ff', type=int, required=True,
                        help='FFN dims')
    parser.add_argument('--ff_dims', type=parse_ff_dims, required=True, 
                        help='Feed-forward layer dimensions as a comma-separated list')
    parser.add_argument('--warmup_ratio', type=float,
                        required=True, help='Warmup ratio for learning rate')
    parser.add_argument('--weight_decay', type=float,
                        required=True, help='Weight decay for AdamW')
    parser.add_argument('--dropout', type=float,
                        required=True, help='Dropout rate')
    parser.add_argument('--src_max_len', type=int, required=True,
                        help='Maximum length of source sequences')
    parser.add_argument('--tgt_max_len', type=int, required=True,
                        help='Maximum length of target sequences')
    parser.add_argument('--curr_epoch', type=int, required=True,
                        help='Current epoch number (used for resuming training)')
    parser.add_argument('--optimizer_lr', type=float,
                        required=True, help='Learning rate for optimizer')
    parser.add_argument('--is_constant_lr', 
                        help='Whether the decay lr be constant',action="store_true")
    parser.add_argument('--is_prefix', 
                        help='Whether the decay lr be constant',action="store_true")
    parser.add_argument('--use_half_precision', 
                        help='Whether to use prefix tokenizer',action="store_true")
    parser.add_argument('--train_shuffle', type=bool, default=False,
                        help='Whether to shuffle training data during each epoch')
    parser.add_argument('--test_shuffle', type=bool,
                        default=False, help='Whether to shuffle test data')
    parser.add_argument('--pin_memory', type=bool, default=False,
                        help='Whether to use pinned memory for data loading')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of processes for distributed training')
    parser.add_argument('--resume_best', type=bool,
                        default=False, help='Whether to resume the best model')
    parser.add_argument('--run_id', type=str, default=None,
                        help='WandB run_id to resume')
    # parser.add_argument('--distributed', type=bool, default=False,
    #                     help='Whether to use distributed training')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='Backend for distributed training')
    parser.add_argument('--src_voc_size', type=int, default=None,
                        help='Size of vocabulary for source sequences')
    parser.add_argument('--tgt_voc_size', type=int, default=None,
                        help='Size of vocabulary for target sequences')
    parser.add_argument('--save_freq', type=int, default=3,
                        help='Epochs at which to save model checkpoints during training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for reproducibility')
    parser.add_argument('--update_lr', type=float,
                        default=None, help='New learning rate')
    parser.add_argument('--end_lr', type=float,
                        default=1e-8, help='End learning rate')
    parser.add_argument('--clip_grad_norm', type=float, default=-1,
                        help='Gradient clipping threshold (set to -1 to disable)')
    parser.add_argument('--save_last', type=bool,
                        default=False, help='Save last model')
    parser.add_argument('--log_freq', type=int, default=50,
                        help='Logging frequency')
    parser.add_argument('--test_freq', type=int, default=10,
                        help='Testing frequency')
    parser.add_argument('--truncate', type=bool, default=False,
                        help='Truncate Sequences')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Run debug mode')
    parser.add_argument('--to_replace', type=bool, default=False,
                        help='Replace index and momentum terms')
    parser.add_argument('--index_pool_size',type=int, default=100,
                        help='Index token pool size')
    parser.add_argument('--momentum_pool_size',type=int, default=100,
                        help='Momentum token pool size')

    args = parser.parse_args()
    return args


def create_config_from_args(args):
    return TransformerConfig(
        project_name=args.project_name,
        run_name=args.run_name,
        model_name=args.model_name,
        root_dir=args.root_dir,
        data_dir=args.data_dir,
        device=args.device,
        epochs=args.epochs,
        training_batch_size=args.training_batch_size,
        test_batch_size=args.test_batch_size,
        valid_batch_size=args.valid_batch_size,
        num_workers=args.num_workers,
        embedding_size=args.embedding_size,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff = args.d_ff,
        ff_dims = args.ff_dims,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        src_max_len=args.src_max_len,
        tgt_max_len=args.tgt_max_len,
        curr_epoch=args.curr_epoch,
        optimizer_lr=args.optimizer_lr,
        is_constant_lr = args.is_constant_lr,
        is_prefix = args.is_prefix,
        use_half_precision=args.use_half_precision,
        train_shuffle=args.train_shuffle,
        test_shuffle=args.test_shuffle,
        pin_memory=args.pin_memory,
        world_size=args.world_size,
        resume_best=args.resume_best,
        run_id=args.run_id,
        # distributed=args.distributed,
        backend=args.backend,
        src_voc_size=args.src_voc_size,
        tgt_voc_size=args.tgt_voc_size,
        save_freq=args.save_freq,
        test_freq = args.test_freq,
        seed=args.seed,
        update_lr=args.update_lr,
        end_lr=args.end_lr,
        clip_grad_norm=args.clip_grad_norm,
        save_last=args.save_last,
        log_freq=args.log_freq,
        debug=args.debug,
        to_replace=args.to_replace,
        index_pool_size=args.index_pool_size,
        momentum_pool_size=args.momentum_pool_size
    )
