from trainer import sequence_accuracy
from fn_utils import create_tokenizer, parse_args, create_config_from_args, init_distributed_mode
from data import Data
import pandas as pd
import numpy as np
import os
import torch

# Parse arguments and create configuration
args = parse_args()
config = create_config_from_args(args)

# Initialize distributed mode
init_distributed_mode(config)

# Get the number of processes and the current process rank
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ["LOCAL_RANK"])

print(config)

# Read train, test, and validation data
df_train = pd.read_csv(config.data_dir + "train.csv")
df_test = pd.read_csv(config.data_dir + "test.csv")
df_valid = pd.read_csv(config.data_dir + "valid.csv")

# Concatenate dataframes for tokenization purposes
df = pd.concat([df_train, df_valid, df_test]).reset_index(drop=True)

# Create tokenizer and vocabularies
tokenizer, src_vocab, tgt_vocab, src_itos, tgt_itos = create_tokenizer(
    df, config, config.index_pool_size, config.momentum_pool_size, is_old=False,
)
config.src_voc_size = len(src_vocab)
config.tgt_voc_size = len(tgt_vocab)

# Split test data among processes
test_splits = [split.reset_index(drop=True) for split in np.array_split(df_test, world_size)]


# Directory to save accuracy files
output_dir = config.root_dir+"/accuracy_outputs/" + config.project_name
os.makedirs(output_dir, exist_ok=True)

# Perform processing for each rank
for i in range(world_size):
    if i == local_rank:
        # Assign test split to current process
        df_test = test_splits[i]
        datasets = Data.get_data(
            df_train, df_test, df_valid, config, tokenizer, src_vocab, tgt_vocab
        )

        test_ds = datasets['test']
        test_accuracy_seq = sequence_accuracy(
            config, test_ds, tgt_itos, True, None, test_size=len(test_ds)
        )

        print(f"SEQUENCE ACCURACY for rank {local_rank}: {test_accuracy_seq}")
        # Save accuracy to a .npy file in the specified output directory
        np.save(os.path.join(output_dir, f"acc_{i}.npy"), np.array(test_accuracy_seq))

# Synchronize all processes
torch.distributed.barrier()

# Compute average accuracy on the main process (rank 0)
if local_rank == 0:
    all_accuracies = []

    # Read accuracies from all files
    for i in range(world_size):
        acc = np.load(os.path.join(output_dir, f"acc_{i}.npy"))
        all_accuracies.append(acc)

    # Compute the average accuracy
    average_accuracy = np.mean(all_accuracies)
    np.save(os.path.join(output_dir, f"average_acc.npy"), np.array(average_accuracy))
    print(f"AVERAGE SEQUENCE ACCURACY: {average_accuracy}")

# Clean up distributed process group
torch.distributed.destroy_process_group()
