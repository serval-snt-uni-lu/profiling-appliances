
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
from os.path import join
import numpy as np
from saxpy import SAX
import os
from pylab import rcParams


rcParams['figure.figsize'] = 15, 8

# ids = [17, 18, 160, 51, 48, 97, 161, 183, 6, 16, 81, 129, 70, 93, 162, 49]
ids = [18, 17, 51, 160, 97, 6, 161, 70, 48, 183]
# ids = [255, 253]
ids = sorted(ids)

root = '/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/'

f = join(root, 'appliance_group_data-2.csv')
annual_data = pd.DataFrame.from_csv(f, index_col=False, header=None)
annual_data.columns = ['interval', 'household', 'appliance', 'date', 'datum', 'time']


d = join(root, 'popular_appliances', 'within_household')

for household in annual_data['household'].unique():
    print household,
    household_data = annual_data[annual_data['household'] == household]
    combd = join(d, '%s' % household)
    if not os.path.exists(combd):
        os.makedirs(combd)
    for i in range(len(ids)):
        idata = household_data[household_data['appliance'] == ids[i]].copy()
        idata = idata.set_index(['date', 'time'])
        del idata['interval']
        del idata['household']
        del idata['appliance']
        if idata.shape[0] < 1:
            continue
        idata.to_csv(join(combd, '%s.csv' % ids[i]))


def create_dataset(csvf, label, l=100, alphabet=10):
    df = pd.DataFrame.from_csv(csvf, index_col=False)
    nrows, _ = df.shape
    cols = range(l+1)
    data = df['datum'].values.tolist()
    ret = []
    for i in range(nrows/l):
        if len(np.unique(data[i*l:(i+1)*l])) < 5:
            continue
        entry = {0: label}
        for j in range(l):
            entry[j+1] = data[i*l+j]
        ret.append(entry)
    adf = pd.DataFrame(ret)
    return adf


from mylib import get_folders

dirs = get_folders('/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/popular_appliances/within_household/')

for root in dirs:
    csvfs = ['%s.csv' % x for x in ids]
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

    t = pd.concat(train, ignore_index=True)
    t.to_csv(join(root, 'train.csv'), index=False)
    t = pd.concat(test, ignore_index=True)
    t.to_csv(join(root, 'test.csv'), index=False)


# In[3]:

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



def saxify_folder(folder, alphabet=(3, 21)):
    print folder
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


def saxify_all(path, alphabet=(3, 21)):
    for folder in get_folders(path):
        saxify_folder(folder, alphabet)
    return


if __name__ == '__main__':
    root = '/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/popular_appliances/within_household/'
    saxify_all(root)


# In[4]:

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
    for i in range(3, 21):
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


if __name__ == '__main__':
    from os.path import join
    root = '/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/'
    build_all_corpora(join(root, 'popular_appliances', 'within_household'))


# In[5]:

import pandas as pd
from os.path import join

from mylib import get_folders

root = '/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/popular_appliances/within_household/'


def generate_parameters(folder, alphabet_range, reset=False, pfile='params.txt'):
    df = pd.DataFrame.from_csv(join(folder, 'train', 'saxified_10.csv'), index_col=False)
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
    generate_parameters(folder, (3, 20))


# In[7]:

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


root = '/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/popular_appliances/within_household/'
for folder in get_folders(root):
    try:
        process_results(folder, resultsdir='final_results_wl_2_to_20')
    except:
        print folder

