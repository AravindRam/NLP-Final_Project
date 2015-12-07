import pandas as pd
from pandas import DataFrame as df
import numpy as np

def main():
    data_train = df.from_csv('train_new.tsv', sep='\t')
    data_test = df.from_csv('public_leaderboard_rel_2.tsv', sep='\t')

    for i in range(1, 11):
        train = data_train[data_train.EssaySet == i]
        test = data_test[data_test.EssaySet == i]
        test.insert(1, 'Score1', 4)
        test.insert(2, 'Score2', 4)
        final = pd.concat([train, test])
        file_name = 'train_test' + str(i) + '.tsv'
        final.to_csv(file_name, sep='\t')
main()