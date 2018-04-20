import os
import re
import nltk
import string
import json
import csv
import argparse
import pandas as pd
import numpy as np
from data import *
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer


ENGLISH_DATA_SMART = '/Users/sneha/Documents/dev/SelfAttentive/autogsr_data/autogsr_smart_interface_english_data_sep16_oct16.json'
ENGLISH_DATA_BASE = '/Users/sneha/Documents/dev/SelfAttentive/autogsr_data/autogsr_data_english_jan16_jul16_translated.json'

EVENT_CLASSES = {u'In-Country Protest Article': 1, u'Out-Country Protest Article':1, u'Non-Protest Article':0}

POPULATION_CLASSES = {u'Agricultural': 8,
 u'Business': 0,
 u'Education': 9,
 u'Ethnic': 6,
 u'General Population': 7,
 u'Labor': 4,
 u'Legal': 3,
 u'Media': 1,
 u'Medical': 2,
 u'Refugees/Displaced': 5,
 u'Religious': 10,
 u'Non-Protest': 11}

PROTEST_CLASSES = {u'Employment and Wages': 0,
 u'Energy and Resources': 4,
 u'Housing': 2,
 u'Other': 3,
 u'Other Economic Policies': 1,
 u'Other Government Policies': 5,
 u'Non-Protest': 6}

# NEW_PROTEST_CLASSES = {u'Other': 0, #3
#                u'Other Government Policies': 1, #5
#                u'Non-Protest': 2 #6
# }
def get_data(DATA_JSON):
    data = []
    key_words = []
    with open(DATA_JSON, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def balance_data(articles):
    balanced_data = []
    positive_cnt = 0
    for article in articles:
        if __get_attribute_label(article, 'eType', EVENT_CLASSES) == '1':
            balanced_data.append(article)
            positive_cnt += 1

    for article in articles:
        while positive_cnt > 0:
            if __get_attribute_label(article, 'eType', EVENT_CLASSES) == '0':
                balanced_data.append(article)
                positive_cnt -= 1

    return balanced_data


def __get_attribute_label(article, attribute, attribute_mapping):
    """

    :param article: the article
    :param attribute_mapping: attribute-class mapping dictionary
    :return:
    """
    cnt = Counter([item[attribute] for item in article['finalEvents']])
    attribute_class = str(attribute_mapping[cnt.most_common()[0][0]]) # get the label with most votes
    return attribute_class


def tokenize(articles):
    #Tokenization and Stemming
    # Remove punctuation and tokenize
    # NO punctuation in text
    stop_words = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
    punctuations = list(string.punctuation )
    Data = []
    cnt = 0
    for a, article in enumerate(articles):
        print(a)
        article_text = article['text']
        event_type = __get_attribute_label(article, 'eType', EVENT_CLASSES)

        if event_type == '1':
            protest_type = __get_attribute_label(article, 'eventType', PROTEST_CLASSES)
            population_type = __get_attribute_label(article, 'populationType', POPULATION_CLASSES)

        else:
            protest_type = str(PROTEST_CLASSES['Non-Protest'])
            population_type = str(POPULATION_CLASSES['Non-Protest'])

        # article_text = re.sub( r'< br/ >', ' ', article_text )
        article_text = article_text.replace('<br/>', ' ')
        # deal with punctuations
        for ch in punctuations:
            article_text = article_text.replace( ch, ' ' )

        tokens = [word for word in word_tokenize( article_text ) if not word in stop_words]
        # TODO: Uncomment for truncating
        #if len(tokens) > 500:
        #    tokens = tokens[:500]
        #    cnt += 1
        # else:
        #     tokens = tokens + ['<PAD>']*(500-len(tokens))

        tagged_tokens = nltk.pos_tag( tokens )

        stemmed = []
        for pair in tagged_tokens:
            # convert verb to its original form
            if pair[ 1 ][:2] == 'VB':
                token = wordnet_lemmatizer.lemmatize(pair[ 0 ], 'v')
            else:
                token = wordnet_lemmatizer.lemmatize(pair[ 0 ])

            stemmed.append(token)

        length = len(stemmed)

        record = [' '.join(stemmed), event_type, protest_type, population_type, str(length)]

        Data.append(record)
    print('Number truncated')
    print(cnt)
    return Data


def data_splits(Data, train_path, valid_path, test_path):
    data_size = len(Data)
    np.random.shuffle(Data)

    Data = [[record[0], record[1], record[2], record[3], int(record[4])] for record in Data]

    # Spilt into train, valid, test - 60%, 20%, 20%
    train_size = int(data_size * 0.6 )
    valid_size = int(data_size * 0.2 )

    train = Data[:train_size ]
    valid = Data[train_size:train_size + valid_size ]
    test = Data[train_size + valid_size:]

    # Sort the reviews in descending order
    train.sort(key=lambda x: x[2], reverse=True )
    valid.sort(key=lambda x: x[2], reverse=True )
    test.sort(key=lambda x: x[2], reverse=True )

    # Save into different files
    split = [train_path, valid_path, test_path]
    for i, data_src in enumerate(split):
        with open(data_src, 'w+' ) as f:
            Writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            if i == 0:
                records = train
            elif i == 1:
                records = valid
            elif i == 2:
                records = test

            records = [[record[0], record[1], record[2], record[3], str(record[4])] for record in records]
            for record in records:
                record = [item for item in record ]
                Writer.writerow(record)


def word_embeddings(data_path, embedding_file, emb_size=200):
    """
    Args:
        embedding_file: Path to the pretrained word vectors
        data_path: Path to store the loaded word vectors for vocab words
    """
    Word2vec_dic = {}
    with open( embedding_file, 'r' ) as f:
        for line in f:
            line = line.split()
            word = line[0]
            vector = line[1:]
            vector = [ float( item ) for item in vector ]
            Word2vec_dic[ word ] = vector


    corpus = Corpus( data_path )
    cuda = False
    emsize = emb_size
    ntokens = len( corpus.dictionary )
    emb_matrix = torch.FloatTensor( ntokens, emsize )
    word_idx_list = []
    initrange = 0.1
    for idx in range( ntokens ):
        try:
            vec = Word2vec_dic[ corpus.dictionary.idx2word[ idx ] ]
            emb_matrix[ idx ] = torch.FloatTensor( vec )
        except:
            word_idx_list.append( idx )
            vec = torch.FloatTensor( 1, emsize )
            vec.uniform_( -initrange, initrange )
            emb_matrix[ idx ] = vec

    # Get Index of Word Embedding that need to be updated during training
    if cuda:
        word_idx_list = torch.cuda.LongTensor( word_idx_list )
    else:
        word_idx_list = torch.LongTensor( word_idx_list )

    torch.save( emb_matrix, 'data/emb_matrix.pt' )
    torch.save( word_idx_list, 'data/word_idx_list.pt' )

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pre-processing for hierarchical Multi-task model')
    parser.add_argument('--data_path', type=str, default='', help='raw data json')
    parser.add_argument('--train_path', type=str, default='data/train.csv', help='location to save train data')
    parser.add_argument('--valid_path', type=str, default='data/valid.csv', help='location to save valid data')
    parser.add_argument('--test_path', type=str, default='data/test.csv', help='location to save test data')
    args = parser.parse_args()

    if args.data_path:
        articles = get_data(args.data_path)
    else:
        articles = get_data(ENGLISH_DATA_SMART) + get_data(ENGLISH_DATA_BASE)
    # print('Balance data....')
    # articles = balance_data(articles)
    print('Start processing data........')
    data = tokenize(articles)
    print('Finished processing data, writing data to files.........')
    data_splits(data, args.train_path, args.valid_path, args.test_path)
    print('Create word embedding file......')
    word_embeddings('/Users/sneha/Documents/dev/SelfAttentive/data/', '/Users/sneha/Documents/dev/SelfAttentive/glove_vectors/glove.twitter.27B.200d.txt', 200)
    print('Done.')
