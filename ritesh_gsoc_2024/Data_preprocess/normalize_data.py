import torchtext; torchtext.disable_torchtext_deprecation_warning()
from tqdm import tqdm
import pandas as pd
import numpy as np
from tokenizer import Tokenizer
import argparse
import random
import os
from sklearn.model_selection import train_test_split

# Argument parsing
parser = argparse.ArgumentParser(description="Augment data parser")
parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
parser.add_argument('--src_file_name', type=str, required=True, help='src file name')
parser.add_argument('--file_name', type=str, required=True, help='Aug file name')
parser.add_argument('--to_replace', type=bool, default=False, help='Replace index and momentum terms')
parser.add_argument('--is_normal', type=bool, default=True, help='Normalize data')
parser.add_argument('--index_pool_size', type=int, default=500, help='Index token pool size')
parser.add_argument('--momentum_pool_size', type=int, default=500, help='Momentum token pool size')
parser.add_argument('--n_samples', type=int, default=1, help='Number of aug samples')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')  # Added seed argument
args = parser.parse_args()

# Constants
BOS_IDX, PAD_IDX, EOS_IDX, UNK_IDX, SEP_IDX = 0, 1, 2, 3, 4
special_symbols = ['<S>', '<PAD>', '</S>', '<UNK>', '<SEP>']

# Set random seed for reproducibility, if provided
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)

# Read the data

df = pd.read_csv(os.path.join(args.data_dir,args.src_file_name))

# Initialize the Tokenizer
tokenizer = Tokenizer(df, args.index_pool_size, args.momentum_pool_size, special_symbols, UNK_IDX,args.to_replace)

def normalize_indices(tokenizer, expressions, index_token_pool_size=50, momentum_token_pool_size=50):
    # Function to replace indices with a new set of tokens for each expression
    def replace_indices(token_list, index_map):
        new_index = (f"INDEX_{i}" for i in range(index_token_pool_size))  # Local generator for new indices
        new_tokens = []
        for token in token_list:
            if "INDEX_" in token:
                if token not in index_map:
                    try:
                        index_map[token] = next(new_index)
                    except StopIteration:
                        # Handle the case where no more indices are available
                        raise ValueError("Ran out of unique indices, increase token_pool_size")
                new_tokens.append(index_map[token])
            else:
                new_tokens.append(token)
        return new_tokens

    def replace_momenta(token_list, index_map):
        new_index = (f"MOMENTUM_{i}" for i in range(momentum_token_pool_size))  # Local generator for new indices
        new_tokens = []
        for token in token_list:
            if "MOMENTUM_" in token:
                if token not in index_map:
                    try:
                        index_map[token] = next(new_index)
                    except StopIteration:
                        # Handle the case where no more indices are available
                        raise ValueError("Ran out of unique indices, increase momentum_token_pool_size")
                new_tokens.append(index_map[token])
            else:
                new_tokens.append(token)
        return new_tokens

    normalized_expressions = []
    # Replace indices in each expression randomly
    for expr in tqdm(expressions,desc="Normalizing.."):
        toks = tokenizer.src_tokenize(expr,42)
        normalized_expressions.append(replace_momenta(replace_indices(toks, {}), {}))

    return normalized_expressions


def aug_data(df):
    # Extract columns
    amps = df['amp']
    sqamps = df['sqamp']

    # Data augmentation
    n_samples = args.n_samples
    aug_amps = []

    for amp in tqdm(amps, desc='processing'):
        random_seed = [random.randint(1, 1000) for _ in range(n_samples)]
        for seed in random_seed:
            aug_amps.append(tokenizer.src_replace(amp, seed))

    aug_sqamps = [sqamp for sqamp in sqamps for _ in range(n_samples)]

    if args.is_normal:
        normal_amps = normalize_indices(tokenizer,aug_amps,args.index_pool_size,args.momentum_pool_size)
        
        aug_amps = []
        for amp in normal_amps:
            aug_amps.append("".join(amp))

    # Create augmented DataFrame
    df_aug = pd.DataFrame({"amp": aug_amps, "sqamp": aug_sqamps})

    return df_aug

df_train, df_valid = train_test_split(
    df, test_size=0.01, random_state=args.seed)

df_train.reset_index(inplace=True, drop=True)

df_valid, df_test = train_test_split(
    df_valid, test_size=0.5, random_state=args.seed)

df_valid.reset_index(inplace=True, drop=True)
df_test.reset_index(inplace=True, drop=True)

del df

data = {'train': df_train ,'test' : df_test, 'valid': df_valid}

for split,df in data.items():
    file_name = args.file_name+split+".csv"
    # Save augmented data
    output_file_path = os.path.join(args.data_dir, file_name)
    df_aug = aug_data(df)
    df_aug.drop_duplicates(inplace=True)
    df_aug.to_csv(output_file_path, index=False)
    print(f'--------{split} done-------')
