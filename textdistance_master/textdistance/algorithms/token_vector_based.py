from .base import Base as _Base, BaseSimilarity as _BaseSimilarity
import numpy as np
from gensim.corpora import dictionary
from numpy import (
    dot, float32 as REAL, double, array, zeros, vstack,
    ndarray, sum as np_sum, prod, argmax, dtype, ascontiguousarray, frombuffer,
)
from pyemd import emd
class WmdDistance(_BaseSimilarity):
    def __init__(self,model,vocab_len):
        self.model = model
        self.vocab_len = vocab_len
    def __call__(self, *sequences):
        assert len(sequences)==2
        document1,document2 = sequences

        def nbow(document):
            d = np.zeros(self.vocab_len, dtype=double)
            nbow = dictionary.doc2bow(document)  # Word frequencies.
            doc_len = len(document)
            for idx, freq in nbow:
                d[idx] = freq / float(doc_len)  # Normalized word frequencies.
            return d

        # Compute nBOW representation of documents. This is what pyemd expects on input.
        d1 = nbow(document1)
        d2 = nbow(document2)

        # Compute WMD.
        return emd(d1, d2, distance_matrix)