from config import TransformerConfig
from model import build_kanformer
from ..tokenizer import Tokenizer
from ..prefix_tokenizer import PrefixTokenizer
import torch.distributed as dist
import torch
import random
from typing import List
import argparse
from datetime import timedelta

from ..constants import BOS_IDX, PAD_IDX, EOS_IDX, UNK_IDX, SPECIAL_SYMBOLS

def create_tokenizer(df, config, index_pool_size, momentum_pool_size):
    """Create a tokenizer and build source and target vocabularies."""
    if config.is_prefix:
        tokenizer = PrefixTokenizer(df, SPECIAL_SYMBOLS, UNK_IDX)
    else:
        tokenizer = Tokenizer(df, index_pool_size, momentum_pool_size, SPECIAL_SYMBOLS, UNK_IDX, config.to_replace)
    
    src_vocab = tokenizer.build_src_vocab(config.seed)
    src_itos = {value: key for key, value in src_vocab.get_stoi().items()}
    tgt_vocab = tokenizer.build_tgt_vocab()
    tgt_itos = {value: key for key, value in tgt_vocab.get_stoi().items()}

    return tokenizer, src_vocab, tgt_vocab, src_itos, tgt_itos

def init_distributed_mode(config):
    """Initialize the distributed processing mode."""
    dist.init_process_group(backend=config.backend, timeout=timedelta(minutes=30))

def generate_unique_random_integers(x, start=0, end=3000):
    """Generate x unique random integers within a given range."""
    if x > (end - start + 1):
        raise ValueError("x cannot be greater than the range of unique values available")
    return random.sample(range(start, end), x)

def decode_sequence(src: List[int], itos):
    """Decode a sequence of token indices into a string."""
    return ''.join(itos[y] for y in src if y not in {PAD_IDX, BOS_IDX, EOS_IDX})

def causal_mask(size):
    """Create a causal mask for a sequence of given size."""
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).int()
    return mask == 0

def calculate_line_params(point1, point2):
    """Calculate the slope and intercept of a line given two points."""
    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2:
        raise ValueError("The x coordinates of the two points must be different to define a straight line.")
    
    m = (y2 - y1) / (x2 - x1)
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
    """Parses command-line arguments for skanformer training."""
    parser = argparse.ArgumentParser(description="Skanformer Training Configuration")

    # Project & Run Details
    parser.add_argument('--project_name', type=str, required=True, help='Project name')
    parser.add_argument('--run_name', type=str, required=True, help='Run identifier')
    parser.add_argument('--model_name', type=str, required=True, help='Model identifier')

    # Directories
    parser.add_argument('--root_dir', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')

    # Hardware & Training Setup
    parser.add_argument('--device', type=str, default='cuda:0', help='Training device (cuda/cpu)')
    parser.add_argument('--epochs', type=int, required=True, help='Total training epochs')
    parser.add_argument('--training_batch_size', type=int, required=True, help='Batch size (training)')
    parser.add_argument('--valid_batch_size', type=int, required=True, help='Batch size (validation)')
    parser.add_argument('--num_workers', type=int, required=True, help='Data loading worker processes')

    # Model Architecture
    parser.add_argument('--embedding_size', type=int, required=True, help='Embedding dimensions')
    parser.add_argument('--nhead', type=int, required=True, help='Transformer attention heads')
    parser.add_argument('--num_layers', type=int, required=True, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, required=True, help='Feed-forward network dimensions')
    parser.add_argument('--ff_dims', type=parse_ff_dims, required=True, help='KAN layer sizes (comma-separated)')

    # Optimization & Regularization
    parser.add_argument('--warmup_ratio', type=float, required=True, help='LR warmup proportion')
    parser.add_argument('--weight_decay', type=float, required=True, help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, required=True, help='Dropout probability')
    parser.add_argument('--optimizer_lr', type=float, required=True, help='Optimizer learning rate')
    parser.add_argument('--is_constant_lr', action='store_true', help='Use constant learning rate')

    # Sequence Configuration
    parser.add_argument('--src_max_len', type=int, required=True, help='Max length (source sequence)')
    parser.add_argument('--tgt_max_len', type=int, required=True, help='Max length (target sequence)')

    # Training Control
    parser.add_argument('--curr_epoch', type=int, required=True, help='Current epoch (resume training)')
    parser.add_argument('--train_shuffle', type=bool, default=False, help='Shuffle training data')
    parser.add_argument('--valid_shuffle', type=bool, default=False, help='Shuffle test data')
    parser.add_argument('--pin_memory', type=bool, default=False, help='Enable pinned memory for dataloader')
    parser.add_argument('--world_size', type=int, default=1, help='Processes for distributed training')
    parser.add_argument('--resume_best', type=bool, default=False, help='Resume best model checkpoint')
    parser.add_argument('--run_id', type=str, default=None, help='Resume run from WandB ID')
    parser.add_argument('--backend', type=str, default='nccl', help='Distributed backend')

    # Miscellaneous
    parser.add_argument('--src_voc_size', type=int, default=None, help='Source vocabulary size')
    parser.add_argument('--tgt_voc_size', type=int, default=None, help='Target vocabulary size')
    parser.add_argument('--save_freq', type=int, default=3, help='Checkpoint save frequency (epochs)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--update_lr', type=float, default=None, help='Updated learning rate')
    parser.add_argument('--end_lr', type=float, default=1e-8, help='Final/Minimum learning rate while linear decay')
    parser.add_argument('--clip_grad_norm', type=float, default=-1, help='Max gradient norm (disable with -1)')
    parser.add_argument('--save_last', type=bool, default=False, help='Save final model checkpoint')
    parser.add_argument('--log_freq', type=int, default=50, help='Logging frequency (steps)')
    parser.add_argument('--test_freq', type=int, default=10, help='Testing frequency (epochs)')
    parser.add_argument('--save_limit', type=int, default=5, help='Max number of saved checkpoints')
    parser.add_argument('--truncate', type=bool, default=False, help='Enable sequence truncation')
    parser.add_argument('--debug', type=bool, default=False, help='Enable debug mode')
    parser.add_argument('--to_replace', type=bool, default=False, help='Replace index/momentum terms')
    parser.add_argument('--index_pool_size', type=int, default=100, help='Index token pool size')
    parser.add_argument('--momentum_pool_size', type=int, default=100, help='Momentum token pool size')

    return parser.parse_args()


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
        valid_shuffle=args.valid_shuffle,
        pin_memory=args.pin_memory,
        world_size=args.world_size,
        resume_best=args.resume_best,
        run_id=args.run_id,
        backend=args.backend,
        src_voc_size=args.src_voc_size,
        tgt_voc_size=args.tgt_voc_size,
        save_freq=args.save_freq,
        save_limit=args.save_limit,
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
