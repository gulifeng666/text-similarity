from gensim import corpora, models, similarities
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
class BaseModel:
    def __init__(self,**config):
        self.config = config
    def load(self):
        pass
    def train(self):
        pass
    def __call__(self, sentence_tokens):
        pass
class Word2Vec(BaseModel):


    def __init__(self,**config):
        super().__init__(**config)
        self.word2vec_path = config['vectors_path']#"e:/data/word2vec/GoogleNews-vectors-negative300.bin"
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_path,binary=True )
        self.config = config
    def train(self):
        raise NotImplementedError
    def __call__(self,sentence_tokens):
        #assert len(sentence_tokens)>=1 and sentence_tokens[0].__class__ == [].__class__
        veclist = []
        wordlist = []
        for token in sentence_tokens:
          try:
            veclist.append(self.word2vec[token.lower()])
            wordlist.append(token.lower())
          except:
            pass
        if(self.config['return_input']==True):
           veclist=veclist+wordlist
        return veclist# for tokens in sentence_tokens]
class Glove(Word2Vec):
      def __init__(self,**config):
          super().__init__(**config)
          self.word2vec_path = config['vectors_path']
      def __int__(self):
          from gensim.test.utils import datapath, get_tmpfile
          from gensim.scripts.glove2word2vec import glove2word2vec
          glove_file = datapath(self.word2vec_path)
          tmp_file = get_tmpfile("test_word2vec.txt")
          glove2word2vec(glove_file, tmp_file)
          self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)
          return self.word2vec
from scipy.sparse import coo_matrix
class GensimModel(BaseModel):
    dic = None
    @classmethod
    def build_dictionary(cls,sentences_tokens):
        cls.dic = corpora.Dictionary(sentences_tokens)
        cls.corpus = [cls.dic.doc2bow(tokens) for tokens in sentences_tokens]
    def __init__(self,name,**config):
        super().__init__(**config)
        if hasattr(models,name):
           self.model =  getattr(models,name)(self.corpus)
        else:
           raise NameError
    def __call__(self,sentence_tokens):
           if(sentence_tokens[0].__class__ != [].__class__):
               sentence_tokens = [sentence_tokens]
           #assert len(sentence_tokens) >= 1 and sentence_tokens[0].__class__ == [].__class__
           sentence_bow = [self.dic.doc2bow(tokens) for tokens in sentence_tokens]
           res = [self.model[bow] for bow in sentence_bow]
           res = [coo_matrix((np.array([item[1] for item in r]), (np.array([0] * len(r)), np.array([item[0] for item in r]))),shape=(1,100)).toarray().T for r in res]
           return np.array(res).reshape(len(res),-1)


# class SentenceEncoder(BaseModel):
#       def
#
#
#
# def run_avg_benchmark(sentences1, sentences2, model=None, use_stoplist=False, doc_freqs=None):
#     if doc_freqs is not None:
#         N = doc_freqs["NUM_DOCS"]
#
#     sims = []
#     for (sent1, sent2) in zip(sentences1, sentences2):
#
#         tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
#         tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens
#
#         tokens1 = [token for token in tokens1 if token in model]
#         tokens2 = [token for token in tokens2 if token in model]
#
#         if len(tokens1) == 0 or len(tokens2) == 0:
#             sims.append(0)
#             continue
#
#         tokfreqs1 = Counter(tokens1)
#         tokfreqs2 = Counter(tokens2)
#         weights1 = [tokfreqs1[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
#                     for token in tokfreqs1] if doc_freqs else None
#         weights2 = [tokfreqs2[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
#                     for token in tokfreqs2] if doc_freqs else None
#
#         embedding1 = np.average([model[token] for token in tokfreqs1], axis=0, weights=weights1).reshape(1, -1)
#         embedding2 = np.average([model[token] for token in tokfreqs2], axis=0, weights=weights2).reshape(1, -1)
#
#         sim = cosine_similarity(embedding1, embedding2)[0][0]
#         sims.append(sim)
#
#     return sims
#
# #word mover's distance
# def run_wmd_benchmark(sentences1, sentences2, model, use_stoplist=False):
#     sims = []
#     for (sent1, sent2) in zip(sentences1, sentences2):
#
#         tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
#         tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens
#
#         tokens1 = [token for token in tokens1 if token in model]
#         tokens2 = [token for token in tokens2 if token in model]
#
#         if len(tokens1) == 0 or len(tokens2) == 0:
#             tokens1 = [token for token in sent1.tokens if token in model]
#             tokens2 = [token for token in sent2.tokens if token in model]
#
#         sims.append(-model.wmdistance(tokens1, tokens2))
#
#     return sims
#
# #sif
# from sklearn.decomposition import TruncatedSVD
#
#
# def remove_first_principal_component(X):
#     svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
#     svd.fit(X)
#     pc = svd.components_
#     XX = X - X.dot(pc.transpose()) * pc
#     return XX
#
#
# def run_sif_benchmark(sentences1, sentences2, model, freqs={}, use_stoplist=False, a=0.001):
#     total_freq = sum(freqs.values())
#
#     embeddings = []
#
#     # SIF requires us to first collect all sentence embeddings and then perform
#     # common component analysis.
#     for (sent1, sent2) in zip(sentences1, sentences2):
#         tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
#         tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens
#
#         tokens1 = [token for token in tokens1 if token in model]
#         tokens2 = [token for token in tokens2 if token in model]
#
#         weights1 = [a / (a + freqs.get(token, 0) / total_freq) for token in tokens1]
#         weights2 = [a / (a + freqs.get(token, 0) / total_freq) for token in tokens2]
#
#         embedding1 = np.average([model[token] for token in tokens1], axis=0, weights=weights1)
#         embedding2 = np.average([model[token] for token in tokens2], axis=0, weights=weights2)
#
#         embeddings.append(embedding1)
#         embeddings.append(embedding2)
#
#     embeddings = remove_first_principal_component(np.array(embeddings))
#     sims = [cosine_similarity(embeddings[idx * 2].reshape(1, -1),
#                               embeddings[idx * 2 + 1].reshape(1, -1))[0][0]
#             for idx in range(int(len(embeddings) / 2))]
#
#     return sims
#
#
# def run_gse_benchmark(sentences1, sentences2,embed):
#     sts_input1 = tf.placeholder(tf.string, shape=(None))
#     sts_input2 = tf.placeholder(tf.string, shape=(None))
#
#     sts_encode1 = tf.nn.l2_normalize(embed(sts_input1))
#     sts_encode2 = tf.nn.l2_normalize(embed(sts_input2))
#
#     sim_scores = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
#
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#
#         [gse_sims] = session.run(
#             [sim_scores],
#             feed_dict={
#                 sts_input1: [sent1.raw for sent1 in sentences1],
#                 sts_input2: [sent2.raw for sent2 in sentences2]
#             })
#     return gse_sims
#
#
# def run_experiment(df, benchmarks):
#     sentences1 = [Sentence(s) for s in df['sent_1']]
#     sentences2 = [Sentence(s) for s in df['sent_2']]
#
#     pearson_cors, spearman_cors = [], []
#     for label, method in benchmarks:
#         sims = method(sentences1, sentences2)
#         pearson_correlation = scipy.stats.pearsonr(sims, df['sim'])[0]
#         print(label, pearson_correlation)
#         pearson_cors.append(pearson_correlation)
#         spearman_correlation = scipy.stats.spearmanr(sims, df['sim'])[0]
#         spearman_cors.append(spearman_correlation)
#     return pearson_cors, spearman_cors
