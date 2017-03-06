"""
copyright Kwangho Heo
2016-12-05
"""
import numpy as np
import argparse
import os

np.random.seed(9742)  # for reproducibility           # acc: 95.6399%
# np.random.seed(6331)  # for reproducibility             # acc: 95.406%

MAX_SEQUENCE_LENGTH = 100
VALIDATION_SPLIT = 0.1


def main():
    parser = argparse.ArgumentParser(description='cnn classifier')
    parser.add_argument('--dataroot', dest='dataroot', action='store', required=True)
    parser.add_argument('--filenm', dest='filenm', action='store', required=True)
    args = parser.parse_args()

    lines = []

    print 'reading tagged file...'
    for i in range(5):
        filenm = '%s%d.txt' % (args.filenm, i+1)
        print 'processing %s' % filenm
        with open(os.path.join(args.dataroot, filenm)) as f:
            lines += f.xreadlines()

    with open('../data/merged_corpus.txt', 'w') as wf:
        wf.writelines(lines)

if __name__ == '__main__':
    main()
