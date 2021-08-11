from .base import Base as _Base, BaseSimilarity as _BaseSimilarity
import numpy as np
from gensim.corpora import dictionary
from numpy import (
    dot, float32 as REAL, double, array, zeros, vstack,
    ndarray, sum as np_sum, prod, argmax, dtype, ascontiguousarray, frombuffer,
)
from functools import reduce
from pyemd import emd
class WmdDistance(_BaseSimilarity):
    def __init__(self):
        pass
        #self.model = model
        #self.vocab_len = vocab_len
    def __call__(self, *sequences):
        assert len(sequences)==2
        document1,document2 = sequences

        docvec1,document1 = document1[:len(document1)//2],document1[len(document1)//2:]
        docvec2,document2 = document2[:len(document2)//2],document2[len(document2)//2:]
        docdict1 = {word:vec for word,vec  in zip(document1,docvec1 )}
        docdict2 = {word:vec for word,vec  in zip(document2,docvec2)}
        docdict1.update(docdict2)
        docdict = {word:0 for word in document1+document2}
        #
        # def nbow(document):
        #     d = np.zeros(self.vocab_len, dtype=double)
        #     nbow = dictionary.doc2bow(document)  # Word frequencies.
        #     doc_len = len(document)
        #     for idx, freq in nbow:
        #         d[idx] = freq / float(doc_len)  # Normalized word frequencies.
        # #     return d
        #
        # # Compute nBOW representation of documents. This is what pyemd expects on input.
        # d1 = nbow(document1)
        # d2 = nbow(document2)
        def f(x,y):
            x[list(y.keys())[0]] = x.get(list(y.keys())[0],0)+1
            return x
        d1 =reduce(f,[docdict.copy()]+[{key:1} for key in document1])
        d2 =reduce(f,[docdict.copy()]+[{key:1} for key in document2])

        # Compute WMD.
        distance_matrix = np.array([[np.sqrt(np.sum((docdict1[list(d1.keys())[i]]-docdict1[list(d2.keys())[j]])**2))for i in range(len(d1))] for j in range(len(d2))] ,dtype='float64')

        return emd(np.array([value/len(d1) for value in list(d1.values())]), np.array([value/len(d2) for value in list(d2.values())]), distance_matrix)