#!/usr/bin/env python
# coding: utf-8

from transformers import AlbertTokenizer, BertTokenizer


def print_from_tokenizer(title, path, sentences, tokenizer_class):
    print('Loading %s tokenizer from %s...' % (title, path))
    tokenizer = tokenizer_class.from_pretrained(path, do_lower_case=False)
    for sentence in sentences:
        print(tokenizer.tokenize(sentence))


sentence1 = 'Qual time da NFL representou a AFC no Super Bowl 50?'
sentence2 = 'A quem a Virgem Maria supostamente apareceu em 1858 em Lourdes, França?'
sentences = [sentence1, sentence2]

print_from_tokenizer(title='Bert', path='bert-base-multilingual-cased', tokenizer_class=BertTokenizer,
                     sentences=sentences)
# print_from_tokenizer(title='Albert', path='/media/discoD/models/sentencepiece/model_unigram_32k',
#                      tokenizer_class=AlbertTokenizer, sentences=sentences)
# print_from_tokenizer(title='Albert', path='/media/discoD/models/sentencepiece/model_bpe_32k',
#                      tokenizer_class=AlbertTokenizer, sentences=sentences)
# print_from_tokenizer(title='Albert', path='/media/discoD/models/sentencepiece/brwac_wiki_eduardo',
#                      tokenizer_class=AlbertTokenizer, sentences=sentences)
# print_from_tokenizer(title='Albert', path='/media/discoD/models/sentencepiece/model_guillou_15k',
#                      tokenizer_class=AlbertTokenizer, sentences=sentences)
# print_from_tokenizer(title='Albert', path='/media/discoD/models/sentencepiece/model_unigram_30k',
#                      tokenizer_class=AlbertTokenizer, sentences=sentences)
print_from_tokenizer(title='Albert', path='/media/discoD/models/sentencepiece/model_unigram_uncased_30k',
                     tokenizer_class=AlbertTokenizer, sentences=sentences)
print_from_tokenizer(title='Albert', path='/media/discoD/models/sentencepiece/model_albert_base_en',
                     tokenizer_class=AlbertTokenizer, sentences=[
        'Yucaipa owned Dominick\'s before selling the chain to Safeway in 1998 for $2.5 billion.',
        'Yucaipa bought Dominick\'s in 1995 for $693 million and sold it to Safeway for $1.8 billion in 1998.'])
