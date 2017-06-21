import os
from glob import glob
import argparse
import numpy as np
from shutil import rmtree, copy2
from utils import get_files

def create_dir(path):
    if os.path.exists(path):
        rmtree(path)
    os.makedirs(path)

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--train_size', type=int)
parser.add_argument('--test_size', type=int)
parser.add_argument('-i', '--input_fname_pattern', default='*.jpg')
args = parser.parse_args()

data_dir = './data'

print('* Loading data')
data = get_files(os.path.join(data_dir, args.dataset),
                 args.input_fname_pattern)
np.random.shuffle(data)

train_dir = os.path.join(data_dir, args.dataset, 'train')
create_dir(train_dir)
test_dir = os.path.join(data_dir, args.dataset, 'test')
create_dir(test_dir)

if args.train_size == None and args.test_size == None:
    args.train_size = int(len(data) * 0.8)
    args.test_size = len(data) - args.train_size
elif args.train_size == None:
    args.train_size = len(data) - args.test_size
elif args.test_size == None:
    args.test_size = len(data) - args.train_size

print('* Copying training set')
for f in data[:args.train_size]:
    copy2(f, train_dir)
print('* Copying test set')
for f in data[args.train_size:args.train_size+args.test_size]:
    copy2(f, test_dir)
print('* Finish')
