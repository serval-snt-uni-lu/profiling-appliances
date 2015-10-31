
# coding: utf-8

# In[ ]:

from mylib import get_folders
import pandas as pd
from os.path import join
import numpy as np


# root = '/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/'

def create_dataset(csvf, label, l=100, alphabet=10):
    df = pd.DataFrame.from_csv(csvf, index_col=False)
    nrows, _ = df.shape
    cols = range(l+1)
    data = df['datum'].values.tolist()
    ret = []
    for i in range(nrows/l):
        if len(np.unique(data[i*l:(i+1)*l])) < 3:
            continue
        entry = {0: label}
        for j in range(l):
            entry[j+1] = data[i*l+j]
        ret.append(entry)
    adf = pd.DataFrame(ret)
    return adf


dirs = get_folders('/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/popular_appliances/combinations/')

for root in dirs:
    csvfs = ["17_18.csv", "17_160.csv", "17_51.csv", "18_160.csv", "18_51.csv", "51_160.csv", "160_161.csv", "17_161.csv", "51_161.csv", "18_183.csv", "6_17.csv", "6_18.csv", "6_51.csv", "6_160.csv", "6_161.csv"]
    train = []
    test = []

    for i in range(len(csvfs)):
        try:
            df = create_dataset(join(root, csvfs[i]), i)
            nrows = df.shape[0]
            train.append(df.iloc[:nrows/2, :])
            test.append(df.iloc[nrows/2:, :])
        except Exception as e:
            pass
#             print e

    if not len(train):
        continue
    t = pd.concat(train, ignore_index=True)
    t.to_csv(join(root, 'train.csv'), index=False)
    t = pd.concat(test, ignore_index=True)
    t.to_csv(join(root, 'test.csv'), index=False)


# In[2]:

from mylib import get_folders
import pandas as pd
from os.path import join


dirs = get_folders('/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/popular_appliances/combinations/')

csvfs = ["17_18.csv", "17_160.csv", "17_51.csv", "18_160.csv", "18_51.csv", "51_160.csv", "160_161.csv", "17_161.csv", "51_161.csv", "18_183.csv", "6_17.csv", "6_18.csv", "6_51.csv", "6_160.csv", "6_161.csv"]

dfs = []
for root in dirs:
    df = pd.DataFrame.from_csv(join(root, 'train.csv'), index_col=False)
    dfs.append(df)
    df = pd.DataFrame.from_csv(join(root, 'test.csv'), index_col=False)
    dfs.append(df)
    
df = pd.concat(dfs, ignore_index=True)
df.to_csv('/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/popular_appliances/combinations_tenfold/all.csv', index=False)


# In[3]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
from os.path import join
import numpy as np
from saxpy import SAX
import os
from pylab import rcParams
import random


root = '/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/'
subd = 'combinations_tenfold'
d = join(root, 'popular_appliances', subd)
if not os.path.exists(d):
    os.makedirs(d)

df = pd.DataFrame.from_csv(join(d, 'all.csv'), index_col=False)
# Ten-fold
total = df.shape[0]
df.index = range(total)
train_sample_size = int(total * 9 / 10)
for i in range(10):
    fdir = join(d, 'fold_%s' % i)
    try:
        os.makedirs(fdir)
    except:
        pass

    train_rows = random.sample(df.index, train_sample_size)
    test_rows = list(set(df.index) - set(train_rows))
    traindf = df.ix[train_rows]
    traindf.to_csv(join(fdir, 'train.csv'), index=False)
    testdf = df.ix[test_rows]
    testdf.to_csv(join(fdir, 'test.csv'), index=False)
        


# In[5]:

from scipy.io import loadmat
import pandas as pd
from saxpy import SAX
from os import listdir
from os.path import isfile, join
import os

from mylib import get_files, get_folders


def saxify_and_export(df, csvf, alphabet=5):
    nrows, ncols = df.shape
    sample_size = ncols - 1
    if sample_size > 100:
        sample_size = 100
    sax = SAX(sample_size, alphabet, 1e-6)
    cols = ['label', 'sax']
    nv = []
    for i in range(nrows):
        values = df.iloc[i, 1:].values.tolist()
        v = {}
        v['label'] = int(df.iloc[i, 0])

        letters, _ = sax.to_letter_rep(values)
        v['sax'] = letters
        nv.append(v)
    return pd.DataFrame(nv, columns=cols).to_csv(csvf, index=False)



def saxify_folder(folder, alphabet=(20, 21)):
    testdir = join(folder, 'test')
    traindir = join(folder, 'train')
    if not os.path.exists(testdir):
        os.makedirs(testdir)
    if not os.path.exists(traindir):
        os.makedirs(traindir)
    for i in range(alphabet[0], alphabet[1]):
        df = pd.DataFrame.from_csv(join(folder, 'train.csv'), index_col=False)
        saxify_and_export(df, join(traindir, 'saxified_%s.csv' % i), alphabet=i)
        df = pd.DataFrame.from_csv(join(folder, 'test.csv'), index_col=False)
        saxify_and_export(df, join(testdir, 'saxified_%s.csv' % i), alphabet=i)
    return


def saxify_all(path, alphabet=(20, 21)):
    for folder in get_folders(path):
        saxify_folder(folder, alphabet)
    return

def saxify_all_multithread(path):
    from multiprocessing import Pool
    pool = Pool(7)
    folders = sorted(get_folders(path))[:3]
    pool.map(saxify_folder, folders)
    return


if __name__ == '__main__':
    saxify_all_multithread(d)


# In[6]:

import numpy as np
import pandas as pd
import os

from collections import defaultdict

from itertools import islice

from mylib import get_folders, get_files

from os import listdir
from os.path import isfile, join
import os


max_word_size = 20

def build_words(seq, n=2):
    # See http://stackoverflow.com/a/7636054
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def build_corpus(sax_csv_file, corpus_dir, min_word_size=2, max_word_size=5):
    df = pd.DataFrame.from_csv(sax_csv_file, index_col=False)
    nrows, ncols = df.shape
    dfs = []
    sizes = []
    for label in df['label'].unique():
        ndf = df[df['label'] == label]
        dfs.append(ndf)
        sizes.append(ndf.shape[0])


    normalize_to = max(sizes)*1000.0
    
    interval = min_word_size + 1
    interval = 1
    cdir = corpus_dir
    if not os.path.exists(cdir):
        try:
            os.makedirs(cdir)
        except:
            pass
    for df in dfs:
        # http://stackoverflow.com/a/17171819
        # rows = np.random.choice(df.index.values, sample_size)
        # df = df.ix[rows]

        l = len(df['sax'].iloc[0])
        c = df['label'].iloc[0]

        dic = defaultdict(int)
        dic2 = defaultdict(int)
        for idx, row in df.iterrows():
            s = row['sax']
            for size in range(min_word_size, max_word_size + 1, interval):
                words = [''.join(x) for x in build_words(s, size)]
                for i in range(len(words)):
                    dic[words[i]] += 1
                for j in range(size):
                    for i in range(j, len(words), size):
                        if i == 0:
                            dic2[('<S>', words[i])] += 1
                        elif i - size > 0:
                            dic2[(words[i-size], words[i])] += 1

        with open('%s/bigram_wl_%s_to_%s_class_%s.txt' % (cdir, min_word_size, max_word_size, c), 'w') as f:
            for k in dic2.keys():
                v = int(dic2[k] * normalize_to / df.shape[0])
                f.write('%s %s\t%s\n' % (k[0], k[1], v))
        with open('%s/unigram_wl_%s_to_%s_class_%s.txt' % (cdir, min_word_size, max_word_size, c), 'w') as f:
            for k in dic.keys():
                v = int(dic[k] * normalize_to / df.shape[0])
                f.write('%s\t%s\n' % (k, v))
    return



def build_corpus_for_folder(folder):
    for i in range(20, 21):
        traincsv = join(folder, 'train', 'saxified_%s.csv' % i)
        cdir = join(folder, 'corpus', 'alphabet_%s' % str(i))
        if not os.path.exists(cdir):
            os.makedirs(cdir)

        print traincsv, cdir
        build_corpus(traincsv, cdir, min_word_size=2, max_word_size=20)
    return


def build_all_corpora(path):
    for folder in get_folders(path):
        build_corpus_for_folder(folder)
    return

def build_all_corpora_multithread(path):
    from multiprocessing import Pool
    pool = Pool(7)
    folders = sorted(get_folders(path))
    pool.map(build_corpus_for_folder, folders)
    for folder in get_folders(path):
        build_corpus_for_folder(folder)
    return

if __name__ == '__main__':
    build_all_corpora_multithread(d)
#     build_all_corpora(d)


# In[ ]:

import pandas as pd
from os.path import join

from mylib import get_folders

root = '/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/popular_appliances/combinations_tenfold/'


def generate_parameters(folder, alphabet_range, reset=False, pfile='params.txt'):
    df = pd.DataFrame.from_csv(join(folder, 'train', 'saxified_20.csv'), index_col=False)
    klasses = sorted(df['label'].unique())
    mod = 'a'
    if reset:
        mod = 'w'
    pfile = join(root, pfile)
    with open(pfile, mod) as f:
        for k in klasses:
            for a in range(alphabet_range[0], alphabet_range[1] + 1):
                f.write('%s %s %s\n' % (folder[len('/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/popular_appliances/'):], a, k))
    return


for folder in get_folders(root):
    generate_parameters(folder, (20, 20))


# In[26]:

root = '/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/popular_appliances/combinations_tenfold/'

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
#             klasses = sorted(traindf['label'].unique())
            klasses = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            df = get_dataframe(folder, resultsdir, a, klasses)
            df = predict(df)
            df.to_csv(clsssifiedf, index=False)
            print folder, len(klasses), a, accuracy_score(df['label'].values.tolist(), df['predicted'].values.tolist())
#             print classification_report(df['label'].values.tolist(), df['predicted'].values.tolist())
        except Exception as e:
            print e
    print
    return


for folder in get_folders(root):
    try:
        process_results(folder, resultsdir='final_results_wl_2_to_20')
    except:
        print folder
        

dfs = []
for folder in get_folders(root):
    f = join(folder, 'final_results_wl_2_to_20', 'alphabet_20', 'classified.csv')
    dfs.append(pd.DataFrame.from_csv(f, index_col=False))
df = pd.concat(dfs, ignore_index=True)
print accuracy_score(df['label'].values.tolist(), df['predicted'].values.tolist())

from sklearn.metrics import confusion_matrix, f1_score
labels = df['label'].unique()
matrix = confusion_matrix(df['label'].values.tolist(), df['predicted'].values.tolist(), labels)
f1 = f1_score(df['label'].values.tolist(), df['predicted'].values.tolist(), average=None)
print labels
print matrix

codes = [r'{\bf Fridge Freezer}', 
         r'{\bf Kettle}', 
         r'{\bf Microwave}', 
         r'{\bf Dishwasher}', 
         r'{\bf Washing Machine}', 
         r'{\bf Shower}', 
         r'{\bf TV LCD}', 
         r'{\bf Light 1}', 
         r'{\bf Light 2}', 
         r'{\bf Vacuum Cleaner}']

print r'\toprule'
for i in range(len(labels)):
    print '&', codes[i],
print '&', r'{\bf F-Measure}', r'\\'
print r'\midrule'
for i in range(len(labels)):
    print codes[i],
    for j in range(len(labels)):
        if i == j:
            print '&', r'{\bf', "{:,}".format(matrix[i][j]), r'}',
        else:
            print '&', "{:,}".format(matrix[i][j]),
    print '&', '%.3f' % f1[i], r'\\'
#     print r'\\'


# In[31]:

from mylib import get_folders
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from os.path import join
from itertools import combinations

def predict(df):
    df['predicted'] = df.iloc[:, 1:].apply(lambda x: int(x.argmax()[len('score_class_'):]), axis=1)
    return df


csvfs = ["17_18.csv", "17_160.csv", "17_51.csv", "18_160.csv", "18_51.csv", "51_160.csv", "160_161.csv", "17_161.csv", "51_161.csv", "18_183.csv", "6_17.csv", "6_18.csv", "6_51.csv", "6_160.csv", "6_161.csv"]
root = '/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/popular_appliances/combinations_tenfold/'

topn = 2

ret = {}

for folder in get_folders(root):
    f = join(folder, 'final_results_wl_2_to_20', 'alphabet_20', 'classified.csv')
    df = pd.DataFrame.from_csv(f, index_col=False)
    labels = combinations([0, 1, 2, 3, 10, 11, 12, 13], 2)
    
    for c in labels:
        n = []
        for i in range(df.shape[0]):
            if df['label'].iloc[i] not in c:
                continue
            x = {'label': df['label'].iloc[i]}
            for l in c:
                x['score_class_%s' % l] = df['score_class_%s' % l].iloc[i]
            n.append(x)
        ndf = pd.DataFrame(n)
        ndf = predict(ndf)
        f1 = join(folder, 'final_results_wl_2_to_20', 'alphabet_20', 'classified_%s_%s.csv' % (c[0], c[1]))
        ndf.to_csv(f1, index=False)

        if '%s %s' % (c[0], c[1]) not in ret.keys():
            ret['%s %s' % (c[0], c[1])] = [ndf]
        else:
            ret['%s %s' % (c[0], c[1])].append(ndf)
#     labels = []
#     print df.groupby(['label']).count().sort(['predicted'], ascending=False).index.tolist()
#     for l in df.groupby(['label']).count().sort(['predicted'], ascending=False).index.tolist():
#         if l in valid_labels:
#             labels.append(l)
#         if len(labels) == topn:
#             break
#     labels = sorted(labels)
#     print labels

#     if len(labels) < topn:
#         continue
#     n = []
#     for i in range(df.shape[0]):
#         if df['label'].iloc[i] not in labels:
#             continue
#         x = {'label': df['label'].iloc[i]}
#         for l in labels:
#             x['score_class_%s' % l] = df['score_class_%s' % l].iloc[i]
#         n.append(x)
#     ndf = pd.DataFrame(n)
#     ndf = predict(ndf)
#     f1 = join(folder, 'final_results_wl_2_to_20', 'alphabet_20', 'classified_%s.csv' % topn)
#     ndf.to_csv(f1, index=False)
#     print folder
#     print labels
# #     print accuracy_score(df['label'].values.tolist(), df['predicted'].values.tolist())
#     print accuracy_score(ndf['label'].values.tolist(), ndf['predicted'].values.tolist())
# #     print confusion_matrix(ndf['label'].values.tolist(), ndf['predicted'].values.tolist())
#     print
#     if sorted(labels) not in ret.keys():
#         ret['%s %s' % (labels[0], labels[1])] = [ndf]
#     else:
#         ret['%s %s' % (labels[0], labels[1])].append(ndf)
csvfs = ["17_18.csv", "17_160.csv", "17_51.csv", "18_160.csv", "18_51.csv", "51_160.csv", "160_161.csv", "17_161.csv", 
         "51_161.csv", "18_183.csv", "6_17.csv", "6_18.csv", "6_51.csv", "6_160.csv", "6_161.csv"]
nmap = {
    '17_18.csv': 'Kettle + Microwave',
    '17_160.csv': 'Kettle + Light 1',
    '17_51.csv': 'Kettle + Washing Machine',
    '18_160.csv': 'Microwave + Light 1',
    '18_51.csv': 'Microwave + Washing Machine',
    '160_161.csv': 'Light 1 + Light 2',
    '17_161.csv': 'Kettle + Light 2',
    '51_161.csv': 'Washing Machine + Light 2',
#     '51_160.csv': 'Washing Machine + Light 1',
    '18_183.csv': 'xxx',
    "6_17.csv": 'Fridge Freezer + Kettle',
    "6_18.csv": 'Fridge Freezer + Microwave',
    "6_51.csv": 'Fridge Freezer + Washing Machine',
    "6_160.csv": 'Fridge Freezer + Light 1',
    "6_161.csv": 'Fridge Freezer + Light 2'
}

for k in ret:
    ndf = pd.concat(ret[k], ignore_index=True)
    labels = sorted(k.split(' '), reverse=True)
    for x in labels:
        print nmap[csvfs[int(x)]], '&',
#     print accuracy_score(ndf['label'].values.tolist(), ndf['predicted'].values.tolist())
    y_true, y_pred = ndf['label'].values.tolist(), ndf['predicted'].values.tolist()
    print '%.3f' % precision_score(y_true, y_pred, pos_label=None, average='weighted'), '&',
    print '%.3f' % recall_score(y_true, y_pred, pos_label=None, average='weighted'), '&',
#     print '%.3f' % f1_score(y_true, y_pred, pos_label=None, average='weighted'), '&',
    print '%.3f' % f1_score(y_true, y_pred, pos_label=None, average='weighted'), r'\\'

