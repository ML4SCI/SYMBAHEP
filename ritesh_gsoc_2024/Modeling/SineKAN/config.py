from dataclasses import dataclass, asdict
from typing import Optional, List

@dataclass
class TransformerConfig:
    """Configuration settings for Skanformer training."""

    # Project & Run Information
    project_name: str
    run_name: str
    model_name: str

    # Directories
    root_dir: str
    data_dir: str

    # Hardware & Training Setup
    device: str
    epochs: int
    training_batch_size: int
    valid_batch_size: int
    num_workers: int

    # Model Architecture
    embedding_size: int
    nhead: int
    num_layers: int
    ff_dims: List[int]
    d_ff: int

    # Optimization & Regularization
    warmup_ratio: float
    dropout: float
    weight_decay: float
    optimizer_lr: float
    is_constant_lr: bool

    # Sequence Configuration
    src_max_len: int
    tgt_max_len: int

    # Training Control
    curr_epoch: int
    is_prefix: bool
    use_half_precision: bool
    train_shuffle: bool
    valid_shuffle: bool
    pin_memory: bool
    world_size: int
    resume_best: bool

    # Optional Parameters
    run_id: Optional[str] = None
    backend: Optional[str] = 'nccl'
    src_voc_size: Optional[int] = None
    tgt_voc_size: Optional[int] = None
    save_freq: Optional[int] = 3
    save_limit: Optional[int] = 3
    seed: Optional[int] = 42
    update_lr: Optional[float] = None
    end_lr: Optional[float] = 1e-8
    clip_grad_norm: Optional[float] = -1
    save_last: Optional[bool] = True
    log_freq: Optional[int] = 50
    test_freq: Optional[int] = 10
    truncate: Optional[bool] = False
    debug: Optional[bool] = False
    to_replace: bool = False
    index_pool_size: int = 100
    momentum_pool_size: int = 100

    def to_dict(self):
        """Convert dataclass to dictionary."""
        return asdict(self)
