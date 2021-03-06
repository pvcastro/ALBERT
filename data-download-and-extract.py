#!/usr/bin/env python3

import configparser
import os
import subprocess
import sys
from urllib.request import urlretrieve

CURDIR = os.path.dirname(os.path.abspath(__file__))
CONFIGPATH = os.path.join(CURDIR, 'config.ini')
config = configparser.ConfigParser()
config.read(CONFIGPATH)

FILEURL = config['DATA']['FILEURL']
FILEPATH = config['DATA']['FILEPATH']
EXTRACTDIR = config['DATA']['TEXTDIR']


def reporthook(blocknum, blocksize, totalsize):
    '''
    Callback function to show progress of file downloading.
    '''
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))


def download():
    if not os.path.exists(FILEPATH):
        urlretrieve(FILEURL, FILEPATH, reporthook)
    else:
        print('%s already downloaded from %s.' % (FILEPATH, FILEURL))


def extract():
    subprocess.call(['python3',
                     os.path.join(CURDIR, 'wikiextractor', 'WikiExtractor.py'),
                     FILEPATH, "--processes=4", "--no_templates", "--filter_disambig_pages", "-o={}".format(EXTRACTDIR)])


def main():
    download()
    extract()


if __name__ == "__main__":
    main()
