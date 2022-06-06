import os
import random
import pickle
import numpy as np
import torch 
import scipy.io
from gensim.models import FastText
from gensim.models.fasttext import FastText as FT_gensim
from tqdm import tqdm


def _fetch(path, name):
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.mat')
        count_file = os.path.join(path, 'bow_tr_counts.mat')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.mat')
        count_file = os.path.join(path, 'bow_va_counts.mat')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.mat')
        count_file = os.path.join(path, 'bow_ts_counts.mat')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.mat')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.mat')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.mat')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.mat')
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts, 
                    'tokens_1': tokens_1, 'counts_1': counts_1, 
                        'tokens_2': tokens_2, 'counts_2': counts_2}
    return {'tokens': tokens, 'counts': counts}

def get_data(path):
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    train = _fetch(path, 'train')
    valid = _fetch(path, 'valid')
    test = _fetch(path, 'test')

    return vocab, train, valid, test

def get_batch(tokens, counts, ind, vocab_size, device, emsize=300):
    """fetch input data by batch."""
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))
    
    for i, doc_id in enumerate(ind):
        doc = tokens[doc_id]
        count = counts[doc_id]
        L = count.shape[1]
        if len(doc) == 1: 
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
    data_batch = torch.from_numpy(data_batch).float().to(device)
    return data_batch

def read_embedding_matrix(vocab, device,  load_trainned):
    """
    read the embedding  matrix passed as parameter and return it as an vocabulary of each word 
    with the corresponding embeddings

    Args:
        path ([type]): [description]

    # we need to use tensorflow embedding lookup heer
    """
    model_path = '/mnt/sdb/home/senior/B10656028/Gensim/test/fasttext_brief.model' # Path.home().joinpath("Projects", 
#                                     "Personal", 
#                                     "balobi_nini", 
#                                     'models', 
#                                     'embeddings_one_gram_fast_tweets_only').__str__()
    embeddings_path = '/mnt/sdb/home/senior/B10656028/Gensim/test/fasttext_brief.model.trainables.vectors_ngrams_lockf.npy' #Path().cwd().joinpath('data', 'preprocess', "embedding_matrix.npy")

    if load_trainned:
        embeddings_matrix = np.load(embeddings_path, allow_pickle=True)
    else:
        model_gensim = FastText.load(model_path)
        vectorized_get_embeddings = np.vectorize(model_gensim.wv.get_vector)
        embeddings_matrix = np.zeros(shape=(len(vocab),300)) #should put the embeding size as a vector
        print("starting getting the word embeddings ++++ ")
        for index, word in tqdm(enumerate(vocab)):
            vector = model_gensim.wv.get_vector(word)
            embeddings_matrix[index] = vector
        print("done getting the word embeddings ")
        with open(embeddings_path, 'wb') as file_path:
            np.save(file_path, embeddings_matrix)

    embeddings = torch.from_numpy(embeddings_matrix).to(device)
    return embeddings