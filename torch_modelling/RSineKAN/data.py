from fn_utils import causal_mask
from torch.utils.data import Dataset
import torch

# Special tokens & coressponding ids
BOS_IDX, PAD_IDX, EOS_IDX, UNK_IDX, SEP_IDX = 0, 1, 2, 3, 4
special_symbols = ['<S>', '<PAD>', '</S>', '<UNK>', '<SEP>']

class Data(Dataset):
    """
    Custom PyTorch dataset for handling data.

    Args:
        df (DataFrame): DataFrame containing data.
    """

    def __init__(self, df, tokenizer, config, src_vocab, tgt_vocab):
        super(Data, self).__init__()
        self.tgt_vals = df['sqamp']
        self.src_vals = df['amp']
        self.tgt_tokenize = tokenizer.tgt_tokenize
        self.src_tokenize = tokenizer.src_tokenize
        self.bos_token = torch.tensor([BOS_IDX], dtype=torch.int64)
        self.eos_token = torch.tensor([EOS_IDX], dtype=torch.int64)
        self.pad_token = torch.tensor([PAD_IDX], dtype=torch.int64)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.config = config

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.src_vals)

    def __getitem__(self, idx):
        """
        Get an item from the dataset at the specified index.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Tuple containing source and target tensors.
        """
        src_tokenized = self.src_tokenize(self.src_vals[idx],self.config.seed)
        tgt_tokenized = self.tgt_tokenize(self.tgt_vals[idx])
        src_ids = self.src_vocab(src_tokenized)
        tgt_ids = self.tgt_vocab(tgt_tokenized)

        enc_num_padding_tokens = self.config.src_max_len - len(src_ids) - 2
        dec_num_padding_tokens = self.config.tgt_max_len - len(tgt_ids) - 1

        if self.config.truncate:
            if enc_num_padding_tokens < 0:
                src_ids = src_ids[:self.config.src_max_len-2]
            if dec_num_padding_tokens < 0:
                tgt_ids = tgt_ids[:self.config.tgt_max_len-1]
        else:
            if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
                raise ValueError("Sentence is too long")

        src_tensor = torch.cat(
            [
                self.bos_token,
                torch.tensor(src_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] *
                             enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        tgt_tensor = torch.cat(
            [
                self.bos_token,
                torch.tensor(tgt_ids, dtype=torch.int64),
                torch.tensor([self.pad_token] *
                             dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(tgt_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        src_mask = (src_tensor != self.pad_token).unsqueeze(0).unsqueeze(0).int() # (1, 1, seq_len)
        tgt_mask = (tgt_tensor != self.pad_token).unsqueeze(0).int() & causal_mask(tgt_tensor.size(0)) # (1, seq_len) & (1, seq_len, seq_len),

        return src_tensor, tgt_tensor, label, src_mask, tgt_mask

    @staticmethod
    def get_data(df_train, df_test, df_valid, config, tokenizer, src_vocab,tgt_vocab):
        """
        Create datasets (train, test, and valid)

        Returns:
            dict: Dictionary containing train, test, and valid datasets.
        """
        train = Data(df_train, tokenizer, config,src_vocab,tgt_vocab)
        test = Data(df_test, tokenizer, config,src_vocab,tgt_vocab)
        valid = Data(df_valid, tokenizer, config,src_vocab,tgt_vocab)

        return {'train': train, 'test': test, 'valid': valid}