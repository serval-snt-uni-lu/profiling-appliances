
# coding: utf-8

# In[6]:

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from os.path import join
import os
import pandas as pd

from mylib import get_folders


def get_dataframe(folder, resultsdir, alphabet, klasses):
    resultsdir = join(folder, resultsdir, 'alphabet_%s' % alphabet)
    dfs = []
    for kls in klasses:
        csvf = join(resultsdir, 'class_%s.csv' % kls)
        df = pd.DataFrame.from_csv(csvf, index_col=False)
        if len(dfs):
            del df['label']
        dfs.append(df)
    return pd.concat(dfs, axis=1)

def predict(df):
    df['predicted'] = df.iloc[:, 1:].apply(lambda x: int(x.argmax()[len('score_class_'):]), axis=1)
    return df

def process_results(folder, resultsdir):
    for a in range(20, 21):
        try:
            clsssifiedf = join(folder, resultsdir, 'alphabet_%s' % a, 'classified.csv')
            traincsv = join(folder, 'train', 'saxified_%s.csv' % a)
            traindf = pd.DataFrame.from_csv(traincsv, index_col=False)
            klasses = sorted(traindf['label'].unique())
            df = get_dataframe(folder, resultsdir, a, klasses)
            df = predict(df)
            df.to_csv(clsssifiedf, index=False)
            print folder, len(klasses), a, accuracy_score(df['label'].values.tolist(), df['predicted'].values.tolist())
#             print classification_report(df['label'].values.tolist(), df['predicted'].values.tolist())
        except Exception as e:
            print e
            pass
    print
    return


root = '/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/popular_appliances/combinations/'
for folder in get_folders(root):
    try:
        process_results(folder, resultsdir='final_results_wl_2_to_20')
    except:
        print folder


# In[ ]:

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from os.path import join
import os
import pandas as pd

from mylib import get_folders


def get_dataframe(folder, resultsdir, alphabet, klasses):
    resultsdir = join(folder, resultsdir, 'alphabet_%s' % alphabet)
    dfs = []
    for kls in klasses:
        csvf = join(resultsdir, 'class_%s.csv' % kls)
        df = pd.DataFrame.from_csv(csvf, index_col=False)
        if len(dfs):
            del df['label']
        dfs.append(df)
    return pd.concat(dfs, axis=1)

def predict(df):
    df['predicted'] = df.iloc[:, 1:].apply(lambda x: int(x.argmax()[len('score_class_'):]), axis=1)
    return df

def process_results(folder, resultsdir):
    for a in range(3, 21):
        try:
            clsssifiedf = join(folder, resultsdir, 'alphabet_%s' % a, 'classified.csv')
            traincsv = join(folder, 'train', 'saxified_%s.csv' % a)
            traindf = pd.DataFrame.from_csv(traincsv, index_col=False)
            klasses = sorted(traindf['label'].unique())
            df = get_dataframe(folder, resultsdir, a, klasses)
            df = predict(df)
            df.to_csv(clsssifiedf, index=False)
            print folder, len(klasses), a, accuracy_score(df['label'].values.tolist(), df['predicted'].values.tolist())
#             print classification_report(df['label'].values.tolist(), df['predicted'].values.tolist())
        except Exception as e:
            print e
    print
    return


root = '/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/popular_appliances/combinations/'
for folder in get_folders(root):
    try:
        process_results(folder, resultsdir='final_results_wl_2_to_20')
    except:
        print folder

