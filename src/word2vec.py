from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec


def vectorizer(
    corpus: List[List[str]], model: Word2Vec, num_features: int = 100
) -> np.ndarray:
    """
    This function takes a list of tokenized text documents (corpus) and a pre-trained
    Word2Vec model as input, and returns a matrix where each row represents the
    vectorized form of a document.

    Args:
        corpus : list
            A list of text documents that needs to be vectorized.

        model : Word2Vec
            A pre-trained Word2Vec model that will be used to vectorize the corpus.

        num_features : int
            The size of the vector representation of each word. Default is 100.

    Returns:
        corpus_vectors : numpy.ndarray
            A 2D numpy array where each row represents the vectorized form of a
            document in the corpus.
    """   

    corpus_vectors = np.zeros((len(corpus), num_features))

    for i, document in enumerate(corpus):
        word_vectors = []
        for word in document:
            if word in model.wv:
                word_vectors.append(model.wv[word])
        
        if word_vectors:
            corpus_vectors[i] = np.mean(word_vectors, axis=0)
        else:
            corpus_vectors[i] = np.zeros(num_features)  

    return corpus_vectors