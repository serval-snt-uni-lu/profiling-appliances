
# coding: utf-8

# In[8]:

import pandas as pd
from os.path import join

from mylib import get_folders

root = '/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/popular_appliances/combinations/'


def generate_parameters(folder, alphabet_range, reset=False, pfile='params.txt'):
    df = pd.DataFrame.from_csv(join(folder, 'train', 'saxified_10.csv'), index_col=False)
    klasses = sorted(df['label'].unique())
    mod = 'a'
    if reset:
        mod = 'w'
    with open(join(root, pfile), mod) as f:
        for k in klasses:
            for a in range(alphabet_range[0], alphabet_range[1] + 1):
                f.write('%s %s %s\n' % (folder[len('/Users/daoyuan.li/Documents/Smart.Buildings/Dataset/DECC/popular_appliances/'):], a, k))
    return


# generate_parameters('NewlyAddedDatasets/ElectricDevices/', (5, 10), reset=True, pfile='params.txt')
# generate_parameters('NewlyAddedDatasets/LargeKitchenAppliances', (3, 20), reset=True, pfile='params1.txt')
# generate_parameters('NewlyAddedDatasets/SmallKitchenAppliances', (3, 20), reset=False, pfile='params1.txt')
# generate_parameters('NewlyAddedDatasets/RefrigerationDevices', (3, 20), reset=False, pfile='params1.txt')

# generate_parameters('NewlyAddedDatasets/ECG5000/', (5, 10), reset=True, pfile='params.txt')
# generate_parameters('Pre_Summer_2015_Datasets/OSULeaf/', (5, 10), reset=True, pfile='params_leaf')
# generate_parameters('Pre_Summer_2015_Datasets/SwedishLeaf/', (5, 10), reset=True, pfile='params_leaf')


for folder in get_folders(root):
    generate_parameters(folder, (20, 20))

