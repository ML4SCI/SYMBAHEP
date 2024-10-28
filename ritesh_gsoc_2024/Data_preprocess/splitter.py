import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


df_train = pd.read_csv("EW_2-to-2train.csv")
df_test = pd.read_csv("EW_2-to-2test.csv")
df_valid = pd.read_csv("EW_2-to-2valid.csv")

df = pd.concat([df_train,df_test,df_valid])
df.reset_index(inplace=True,drop=True)

src_arr = np.load("src_arr.npy")
tgt_arr = np.load("tgt_arr.npy")

indices = []
for i in range(len(src_arr)):
    if((src_arr[i] <= 400) and (tgt_arr[i] <= 1200)):
        indices.append(i)

df_new = df.iloc[indices].copy()

del df

df_train, df_valid = train_test_split(
    df_new, test_size=5000)

df_train.reset_index(inplace=True, drop=True)

df_valid, df_test = train_test_split(
    df_valid, test_size=0.5)

df_valid.reset_index(inplace=True, drop=True)
df_test.reset_index(inplace=True, drop=True)

print(df_train.shape,df_valid.shape,df_test.shape)

df_train.to_csv("EW_2-to-2train.csv",index=False)
df_test.to_csv("EW_2-to-2test.csv",index=False)
df_valid.to_csv("EW_2-to-2valid.csv",index=False)