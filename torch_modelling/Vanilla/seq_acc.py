from trainer import sequence_accuracy
from fn_utils import create_tokenizer, parse_args,create_config_from_args
from data import Data
import pandas as pd


args = parse_args()
config = create_config_from_args(args)

print(config)

df_train = pd.read_csv(config.data_dir+"train.csv")
df_test = pd.read_csv(config.data_dir+"test.csv")
df_valid = pd.read_csv(config.data_dir+"valid.csv")

df = pd.concat([df_train,df_valid,df_test]).reset_index(drop=True)

tokenizer, src_vocab, tgt_vocab, src_itos, tgt_itos = create_tokenizer(df,config,
                                                                           config.index_pool_size,config.momentum_pool_size)
config.src_voc_size = len(src_vocab)
config.tgt_voc_size = len(tgt_vocab)



datasets = Data.get_data(
            df_train, df_test, df_valid, config, tokenizer,src_vocab, tgt_vocab)

test_ds = datasets['test']

test_accuracy_seq = sequence_accuracy(config,test_ds,tgt_itos,True, None,test_size=len(test_ds))

print(f"SEQUENCE ACCURACY : {test_accuracy_seq}")