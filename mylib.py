
# coding: utf-8

# In[ ]:

from os import listdir
from os.path import isfile, join
import os


def get_folders(path):
    return [join(path, f) for f in listdir(path) if not isfile(join(path, f))]

def get_files(path, ft='.mat'):
    return [join(path, f) for f in listdir(path) if f.endswith(ft) and isfile(join(path, f))]

