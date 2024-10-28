from prefix_tokenizer import PrefixTokenizer
from config import TransformerConfig
from model import Model
from tokenizer import Tokenizer
import torch.distributed as dist
import torch
import torch.nn as nn
import random
from typing import List
from torch.nn.utils.rnn import pad_sequence
import argparse
from datetime import timedelta

# Special tokens & coressponding ids
BOS_IDX, PAD_IDX, EOS_IDX, UNK_IDX, SEP_IDX = 0, 1, 2, 3, 4
special_symbols = ['<S>', '<PAD>', '</S>', '<UNK>', '<SEP>']


def create_tokenizer(df, config, index_pool_size, momentum_pool_size,is_old=False):
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


def generate_eqn_mask(n: int, device: torch.device) -> torch.Tensor:
    """
    Generate an equation mask for the Transformer model.

    Args:
        n (int): The size of the mask.
        device (torch.device): The device on which to create the mask.

    Returns:
        torch.Tensor: The equation mask.
    """
    mask = (torch.triu(torch.ones((n, n), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src: torch.Tensor, tgt: torch.Tensor, device: torch.device) -> tuple:
    """
    Create masks for source and target sequences.

    Args:
        src (torch.Tensor): Source sequence.
        tgt (torch.Tensor): Target sequence.
        device (torch.device): Device on which to create the masks.

    Returns:
        tuple: Tuple containing four masks: source mask, target mask, source padding mask, target padding mask.
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # Generate equation mask for target sequence
    tgt_mask = generate_eqn_mask(tgt_seq_len, device)

    # Create source mask
    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           device=device).type(torch.bool)

    # Create source and target padding masks
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


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


def collate_fn(batch: list) -> tuple:
    """
    Collate function for batching sequences.

    Args:
        batch (list): List of tuples containing source and target sequences.

    Returns:
        tuple: Tuple containing padded source batch and padded target batch.
    """
    src_batch, tgt_batch = [], []
    for (src_sample, tgt_sample) in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    # Pad sequences in the batch
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

    return src_batch, tgt_batch


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
    model = Model(config.num_encoder_layers, config.num_decoder_layers, config.embedding_size,
                  config.nhead, config.src_voc_size, config.tgt_voc_size, config.hidden_dim, config.dropout)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


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
    parser.add_argument('--device', type=str, default='cuda',
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
    parser.add_argument('--hidden_dim', type=int, required=True,
                        help='Dimensionality of hidden layers in the transformer model')
    parser.add_argument('--nhead', type=int, required=True,
                        help='Number of attention heads in the transformer model')
    parser.add_argument('--num_encoder_layers', type=int, required=True,
                        help='Number of encoder layers in the transformer model')
    parser.add_argument('--num_decoder_layers', type=int, required=True,
                        help='Number of decoder layers in the transformer model')
    parser.add_argument('--warmup_ratio', type=float,
                        required=True, help='Warmup ratio for learning rate')
    parser.add_argument('--dropout', type=float,
                        required=True, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float,
                        required=True, help='Weight decay for AdamW')
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
    parser.add_argument('--use_half_precision', 
                        help='Whether to use half precision (FP16) for training',action="store_true")
    parser.add_argument('--is_prefix', 
                        help='Whether to use half prefix expressions',action="store_true")
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
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        src_max_len=args.src_max_len,
        tgt_max_len=args.tgt_max_len,
        curr_epoch=args.curr_epoch,
        optimizer_lr=args.optimizer_lr,
        is_constant_lr = args.is_constant_lr,
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
        is_prefix=args.is_prefix,
        index_pool_size=args.index_pool_size,
        momentum_pool_size=args.momentum_pool_size
    )
