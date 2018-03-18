import os
import re
import nltk
import string
import json
import csv
import pandas as pd
import numpy as np
from data import *
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer


ENGLISH_DATA_SMART = '/Users/sneha/Documents/dev/SelfAttentive/autogsr_data/autogsr_smart_interface_english_data_sep16_oct16.json'
ENGLISH_DATA_BASE = '/Users/sneha/Documents/dev/SelfAttentive/autogsr_data/autogsr_data_english_jan16_jul16_translated.json'

EVENT_CLASSES = {u'In-Country Protest Article': 1, u'Out-Country Protest Article':1, u'Non-Protest Article':0}

def get_data(DATA_JSON):
    data = []
    key_words = []
    with open(DATA_JSON, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def tokenize(articles):
    #Tokenization and Stemming
    # Remove punctuation and tokenize

    # NO punctuation in text
    wordnet_lemmatizer = WordNetLemmatizer()
    punctuations = list( string.punctuation )

    Data = []
    for article in articles:
        article_text = article['text']
        cnt = Counter([item['eType'] for item in article['finalEvents']])
        event_type = str(EVENT_CLASSES[cnt.most_common()[0][0]]) # get the label with most votes

        # article_text = re.sub( r'< br/ >', ' ', article_text )
        article_text = article_text.replace('<br/>', ' ')
        # deal with punctuations
        for ch in punctuations:
            article_text = article_text.replace( ch, ch + ' ' )

        tokens = [word for word in word_tokenize( article_text ) ]
        tagged_tokens = nltk.pos_tag( tokens )

        stemmed = []
        for pair in tagged_tokens:
            # convert verb to its original form
            if pair[ 1 ][:2] == 'VB':
                token = wordnet_lemmatizer.lemmatize( pair[ 0 ], 'v' )
            else:
                token = wordnet_lemmatizer.lemmatize( pair[ 0 ] )
                
            stemmed.append( token )

        length = len( stemmed )
        record = [ ' '.join( stemmed ), event_type, str( length ) ]
        Data.append( record )

    return Data

def data_splits(Data):
    data_size = len( Data )
    np.random.shuffle( Data )

    Data = [ [record[0], record[1], int( record[2]) ] for record in Data ]

    # Spilt into train, valid, test - 60%, 20%, 20%
    train_size = int( data_size * 0.6 )
    valid_size = int( data_size * 0.2 )

    train = Data[ :train_size ]
    valid = Data[ train_size:train_size + valid_size ]
    test = Data[ train_size + valid_size:]

    # Sort the reviews in descending order
    train.sort( key=lambda x:x[2], reverse=True )
    valid.sort( key=lambda x:x[2], reverse=True )
    test.sort( key=lambda x:x[2], reverse=True )

    # Save into different files
    split = [ 'train', 'valid', 'test' ]
    for data_src in split:
        with open(  'data/' + data_src + '.csv', 'w+' ) as f:
            Writer = csv.writer( f, delimiter=',', quoting=csv.QUOTE_MINIMAL )
            
            if data_src == 'train':
                Records = train
            elif data_src == 'valid':
                Records = valid
            else:
                Records = test
                
            Records = [ [record[0], record[1], str(record[2])] for record in Records ]
                
            for record in Records:
                # record = [ item.encode('utf-8') for item in record ]
                Writer.writerow( record )


    
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
    articles = get_data(ENGLISH_DATA_SMART) + get_data(ENGLISH_DATA_BASE)
    # articles = get_data(ENGLISH_DATA_SMART)
    print('Start processing data........')
    data = tokenize(articles)
    print('Finishes processing data, writing data to files.........')
    data_splits(data)
    print('Create word embedding file......')
    word_embeddings('/Users/sneha/Documents/dev/SelfAttentive/data/', '/Users/sneha/Documents/dev/SelfAttentive/glove_vectors/glove.twitter.27B.200d.txt', 200)
    print('Done.')
