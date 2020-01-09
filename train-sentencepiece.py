#!/usr/bin/env python3

import configparser
import glob
import os
import sentencepiece as sp

CURDIR = os.path.dirname(os.path.abspath(__file__))
CONFIGPATH = os.path.join(CURDIR, 'config.ini')
config = configparser.ConfigParser()
config.read(CONFIGPATH)

TEXTDIR = config['DATA']['TEXTDIR']
PREFIX = config['SENTENCEPIECE']['PREFIX']
VOCABSIZE = config['SENTENCEPIECE']['VOCABSIZE']
CTLSYMBOLS = config['SENTENCEPIECE']['CTLSYMBOLS']


def _get_text_file(text_dir=TEXTDIR):
    file_list = glob.glob(f'{text_dir}/**/all_sp.txt')
    files = ",".join(file_list)
    return files

#CTLSYMBOLS = [CLS],[SEP],[MASK]

def train(prefix=PREFIX, vocab_size=VOCABSIZE, ctl_symbols=CTLSYMBOLS):
    files = _get_text_file()
    # command = f'--input={files} --model_prefix={prefix} --vocab_size={vocab_size} --control_symbols={ctl_symbols}'
    command = f'--input={files} --model_prefix={prefix} --vocab_size={vocab_size} --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --pad_piece=<pad> --unk_piece=<unk> --bos_piece=[CLS] --eos_piece=[SEP] --control_symbols={ctl_symbols} --user_defined_symbols=(,),\",-,.,–,£,€,$'
    # command = f'--input={files}  --vocab_size={vocab_size} --model_prefix={prefix} --pad_id=0 --unk_id=1 --pad_piece=<pad> --unk_piece=<unk> --bos_id=-1 --eos_id=-1 --control_symbols=[CLS],[SEP],[MASK],<pad>'
    # command = f'--input={files}  --vocab_size={vocab_size} --model_prefix={prefix} --model_type=bpe --pad_id=0 --unk_id=1 --pad_piece=<pad> --unk_piece=<unk> --bos_id=-1 --eos_id=-1 --control_symbols=[CLS],[SEP],[MASK],<pad>'
    # command = f'--input={files}  --vocab_size={vocab_size} --model_prefix={prefix} --model_type=bpe --bos_id=-1 --eos_id=-1 --control_symbols={ctl_symbols}'
    # command = f'--input={files}  --vocab_size={vocab_size} --model_prefix={prefix} --model_type=bpe --control_symbols={ctl_symbols}'
    # command = f'--input={files} --model_prefix={prefix} --vocab_size={vocab_size} --control_symbols={ctl_symbols}'
    sp.SentencePieceTrainer.Train(command)


def main():
    train()


if __name__ == "__main__":
    main()
