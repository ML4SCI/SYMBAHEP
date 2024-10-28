from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class TransformerConfig:

    # Project & Run name
    project_name: str
    run_name: str

    # Model name
    model_name: str

    # Directory where data and model checkpoints will be stored
    root_dir: str

    data_dir: str
    # Device for training (e.g., "cuda" for GPU, "cpu")
    device: str

    # Total number of epochs for training
    epochs: int

    # Batch size for training
    training_batch_size: int

    # Batch size for testing
    test_batch_size: int

    valid_batch_size: int

    # Number of worker processes for data loading
    num_workers: int

    # Dimensionality of word embeddings
    embedding_size: int

    # Dimensionality of hidden layers in the transformer model
    hidden_dim: int

    # Number of attention heads in the transformer model
    nhead: int

    # Number of encoder layers in the transformer model
    num_encoder_layers: int

    # Number of decoder layers in the transformer model
    num_decoder_layers: int

    # Warmup ratio for learning rate
    warmup_ratio: float

    # Dropout rate
    dropout: float

    # Weight decay
    weight_decay: float
    
    # Maximum length of source and target sequences
    src_max_len: int
    tgt_max_len: int

    # Current epoch number (used for resuming training)
    curr_epoch: int

    # Learning rate for optimizer
    optimizer_lr: float

    # Decay lr type
    is_constant_lr: bool

    # Whether to use half precision (FP16) for training
    use_half_precision: bool

    # Whether to shuffle training data during each epoch
    train_shuffle: bool

    # Whether to shuffle test data
    test_shuffle: bool

    # Whether to use pinned memory for data loading (faster on GPU)
    pin_memory: bool

    world_size: int

    resume_best: bool
    # WandB run_id to resume
    run_id: Optional[str] = None

    # # distributed training
    # distributed: Optional[bool] = True

    backend: Optional[str] = 'nccl'

    # Size of vocabulary for source and target sequences
    src_voc_size: Optional[int] = None
    tgt_voc_size: Optional[int] = None

    # Epochs at which to save model checkpoints during training
    save_freq: Optional[int] = 3

    # Seed for reproducibility
    seed: Optional[int] = 42

    # New learning rate
    update_lr: Optional[float] = None

    # End learning rate
    end_lr: Optional[float] = 1e-8

    # Gradient clipping threshold (set to -1 to disable)
    clip_grad_norm: Optional[float] = -1

    # Save last model
    save_last: Optional[bool] = True

    # Logging frequency
    log_freq: Optional[int] = 50

    # test frequency
    test_freq: Optional[int] = 10

    # trucate sequences
    truncate: Optional[bool]= False

    # if debug
    debug: Optional[bool] = False
    
    #to replace index and momentum
    to_replace: bool = False

    #to replace index and momentum
    is_prefix: bool = False

    #token pool sizes
    index_pool_size : int = 100   
    momentum_pool_size : int = 100

    def to_dict(self):
        return asdict(self)
