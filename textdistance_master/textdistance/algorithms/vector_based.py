"""
IMPORTANT: it's just draft
"""
# built-in
from functools import reduce

# app
from .base import Base as _Base, BaseSimilarity as _BaseSimilarity


try:
    import numpy
except ImportError:
    numpy = None


class Chebyshev(_Base):
    def _numpy(self, s1, s2):
        s1, s2 = numpy.asarray(s1), numpy.asarray(s2)
        return numpy.max(numpy.abs(s1 - s2))

    def _pure(self, s1, s2):
        return numpy.max(numpy.abs(e1 - e2) for e1, e2 in zip(s1, s2))

    def __call__(self, s1, s2):
        if numpy:
            return self._numpy(s1, s2)
        else:
            return self._pure(s1, s2)


class Minkowski(_Base):
    def __init__(self, p=1, weight=1):
        if p < 1:
            raise ValueError('p must be at least 1')
        self.p = p
        self.weight = weight

    def _numpy(self, s1, s2):
        s1, s2 = numpy.asarray(s1), numpy.asarray(s2)
        result = (self.weight * abs(s1 - s2)) ** self.p
        return result.sum() ** (1.0 / self.p)

    def _pure(self, s1, s2):
        result = (self.weight * abs(e1 - e2) for e1, e2 in zip(s1, s2))
        result = sum(e ** self.p for e in result)
        return result ** (1.0 / self.p)

    def __call__(self, s1, s2):
        if numpy:
            return self._numpy(s1, s2)
        else:
            return self._pure(s1, s2)


class Manhattan(_Base):
    def __call__(self, s1, s2):
        raise NotImplementedError

from sklearn.metrics.pairwise import paired_cosine_distances
class VectorCosine(_Base):
    def __init__(self, squared=False):
        self.squared = squared

    def _numpy(self, s1, s2):
        s1 = numpy.asarray(s1)

        s2 = numpy.asarray(s2)
        return paired_cosine_distances(s1.reshape(1,-1),s2.reshape(1,-1))[0]

    def _pure(self, s1, s2):
        raise NotImplementedError

    def __call__(self, s1, s2):
        if numpy:
            return self._numpy(s1, s2)
class Euclidean(_Base):
    def __init__(self, squared=False):
        self.squared = squared

    def _numpy(self, s1, s2):
        s1 = numpy.asarray(s1)
        s2 = numpy.asarray(s2)
        q = numpy.matrix(s1 - s2)
        result = (q * q.T).sum()
        if self.squared:
            return result
        return numpy.sqrt(result)

    def _pure(self, s1, s2):
        raise NotImplementedError

    def __call__(self, s1, s2):
        if numpy:
            return self._numpy(s1, s2)
        else:
            return self._pure(s1, s2)


class Mahalanobis(_Base):
    def __call__(self, s1, s2):
        raise NotImplementedError


class Correlation(_BaseSimilarity):
    def _numpy(self, *sequences):
        sequences = [numpy.asarray(s) for s in sequences]
        ssm = [s - s.mean() for s in sequences]
        result = reduce(numpy.dot, sequences)
        for sm in ssm:
            result /= numpy.sqrt(numpy.dot(sm, sm))
        return result

    def _pure(self, *sequences):
        raise NotImplementedError

    def __call__(self, *sequences):
        if numpy:
            return self._numpy(*sequences)
        else:
            return self._pure(*sequences)


class Kulsinski(_BaseSimilarity):
    def __call__(self, s1, s2):
        raise NotImplementedError
