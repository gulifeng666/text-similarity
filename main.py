import pandas as pd
import numpy as np
import scipy
import math
import os
#import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gensim.models as gsm
import yaml
import seaborn
from tqdm import tqdm
import nltk
import textdistance_master.textdistance as td
from transformers.pipelines import Pipeline
from transformers import AutoModel,AutoConfig,AutoTokenizer,AutoFeatureExtractor,AutoModelForTokenClassification,AutoModelWithLMHead
import gensim

def load_sts_dataset(filename):
    # Loads a subset of the STS dataset into a DataFrame. In particular both
    # sentences and their human rated similarity score.
    sent_pairs = []
    with open(filename,encoding='utf-8',mode = "r") as f:
        for line in f:
            ts = line.strip().split("\t")
            sent_pairs.append((ts[5], ts[6], float(ts[4])))
    return pd.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])
sts_dev = load_sts_dataset(os.path.join("data/stsbenchmark", "sts-dev.csv"))
sts_test = load_sts_dataset(os.path.join("data/stsbenchmark", "sts-test.csv"))
# import requests
def load_sick(file):
    with open(file) as f:
        lines = f.readlines()[1:]
        lines = [l.split("\t") for l in lines if len(l) > 0]
        lines = [l for l in lines if len(l) == 5]
        df = pd.DataFrame(lines, columns=["idx", "sent_1", "sent_2", "sim", "label"])
        df['sim'] = pd.to_numeric(df['sim'])
        return df
sick_train =  load_sick(os.path.join("e:data", "SICK_train.txt"))
sick_dev  =  load_sick(os.path.join("e:data", "SICK_trial.txt"))
sick_test  =  load_sick(os.path.join("e:data", "SICK_test_annotated.txt"))
from  dataloader.dataloader import DataLoader
from models.model import Word2Vec,GensimModel
config = yaml.load(open('config/config.yaml'))
data_loader = DataLoader()
data_train,data_dev,data_test = data_loader.load_sick()
data_train =data_train[:10]
data_dev = data_dev[:10]
data_test = data_test[:10]
#word2vec = Word2Vec(config['models']['word2vec_model'])
#gensim_models = [GensimModel(config['gensim_model']['model'+str(i)] for i in range(len(list(filter(lambda x:'model' in x,config['gensim_model'].keys())))))]
GensimModel.build_dictionary([nltk.word_tokenize(sentence) for sentence in data_train['sent_1'].tolist()+data_train['sent_2'].tolist()+data_dev['sent_1'].tolist()+data_dev['sent_2'].tolist()])
dataframe = pd.DataFrame(index = ['pearson','spearmanr'])
for pipeline_key,pipeline_value in config['pipeline'].items():
    for preprocess_name in map(lambda x:x.strip(),pipeline_value['preprocess'].split(',')):
        if(preprocess_name=='None'):
            preprocess = lambda x:x
        elif(preprocess_name == 'move_stopword'):
            preprocess = lambda x:x
        elif(preprocess_name == 'tokenlize'):
            preprocess = lambda x:nltk.word_tokenize(x)
        else:
            raise NameError
        from scipy.sparse import coo_matrix
        for model_key in map(lambda x:x.strip(),pipeline_value['model'].keys()):
            model_numbers = map(lambda x: x.strip(), pipeline_value['model'][model_key].split(',')) if model_key != 'None' else [None]
            for model_number in model_numbers:
                if(model_key=='None'):
                    model = lambda x :x
                elif(model_key=='gensim_model'):
                    model = GensimModel(config['models'][model_key][model_number]['name'])
                elif(model_key=='transformer_model'):
                    transformer_model = AutoModel.from_pretrained(config['models'][model_key][model_number]['name'])#bert-base-uncased')
                    tokenizer = AutoTokenizer.from_pretrained(config['models'][model_key][model_number]['name'])
                    #model(**tokenizer('i want', return_tensors='pt'))
                    model = lambda x:transformer_model(**tokenizer(x,return_tensors='pt'))[0]
                elif(model_key == 'word2vec_model'):
                    model = Word2Vec(**config['models'][model_key][model_number])
                else:
                    raise NameError
                for postprocess_name in map(lambda x:x.strip(),pipeline_value['postprocess'].split(',')) :
                    if(postprocess_name == 'None'):
                        postprocess = lambda x:x
                    elif(postprocess_name=='Avg'):
                        postprocess = lambda x:np.mean(x.detach().numpy() if type(x)==torch.Tensor else x,-2)
                    elif(postprocess_name=='MaxPool'):
                        postprocess = lambda x:np.max(x.detach().numpy() if type(x)==torch.Tensor else x,-2)
                    else:
                      raise NameError
                    for sim_method in tqdm(map(lambda x: x.strip(), pipeline_value['simillarity'].keys())):
                        for sim_method_name in  tqdm(map(lambda x: x.strip(),config['simillarity'][sim_method].split(','))):
                           sim = getattr(td, sim_method_name)() if 'WMD' not in sim_method_name else getattr(td, sim_method_name)(model)
                           sim = sim.similarity if 'vector' not in sim_method else sim
                           res = [sim(postprocess(model(preprocess(sentencepair[0] ))),postprocess(model(preprocess(sentencepair[1])))) for sentencepair in zip(data_test["sent_1"].tolist(),data_test["sent_2"].tolist())]
                           res = list(map(lambda x:-x,res)) if 'vector'  in sim_method else res
                           pearson_correlation = scipy.stats.pearsonr(res,data_test['sim'].tolist())[0]
                           spearmanr_correlation = scipy.stats.spearmanr(res,data_test['sim'].tolist())[0]
                           dataframe [preprocess_name+model_key+str(model_number)+sim_method+sim_method_name] = [pearson_correlation,spearmanr_correlation]
                           plt.rcParams['font.size'] = 5

                           fig = dataframe.T.plot.bar()

                           plt.tight_layout()
                           plt.show()

# sick_train = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_train.txt")
# sick_dev = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_trial.txt")
# sick_test = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_test_annotated.txt")
# sick_all = sick_train.append(sick_test).append(sick_dev)
import math
from gensim import corpora, models, similarities
import nltk
sick_train_list = sick_train['sent_1'].to_list()+sick_train['sent_2'].to_list()
sick_train_tokens = [nltk.word_tokenize(sick_train_text) for sick_train_text in sick_train_list]
dic = corpora.Dictionary(sick_train_tokens)
corpus = [dic.doc2bow(text) for text in sick_train_tokens]
tfidf=models.LdaModel(corpus)
sim_matrix = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=len(dic.token2id))
sim = sim_matrix[tfidf[corpus[0]]]

import csv

PATH_TO_FREQUENCIES_FILE = "data/frequencies.tsv"
PATH_TO_DOC_FREQUENCIES_FILE = "data/doc_frequencies.tsv"
def read_tsv(f):
    frequencies = {}
    with open(f) as tsv:
        tsv_reader = csv.reader(tsv, delimiter="\t")
        for row in tsv_reader:
            frequencies[row[0]] = int(row[1])

    return frequencies
frequencies = read_tsv(PATH_TO_FREQUENCIES_FILE)
doc_frequencies = read_tsv(PATH_TO_DOC_FREQUENCIES_FILE)
doc_frequencies["NUM_DOCS"] = 1288431


import nltk
#nltk.download()
STOP = set(nltk.corpus.stopwords.words("english"))
class Sentence:

    def __init__(self, sentence):
        self.raw = sentence
        normalized_sentence = sentence.replace("‘", "'").replace("’", "'")
        self.tokens = [t.lower() for t in nltk.word_tokenize(normalized_sentence)]
        self.tokens_without_stop = [t for t in self.tokens if t not in STOP]

import gensim
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec

word2vec_path = "e:/data/word2vec/GoogleNews-vectors-negative300.bin"

word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter



#向量平均
def run_avg_benchmark(sentences1, sentences2, model=None, use_stoplist=False, doc_freqs=None):
    if doc_freqs is not None:
        N = doc_freqs["NUM_DOCS"]

    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):

        tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
        tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

        tokens1 = [token for token in tokens1 if token in model]
        tokens2 = [token for token in tokens2 if token in model]

        if len(tokens1) == 0 or len(tokens2) == 0:
            sims.append(0)
            continue

        tokfreqs1 = Counter(tokens1)
        tokfreqs2 = Counter(tokens2)
        weights1 = [tokfreqs1[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                    for token in tokfreqs1] if doc_freqs else None
        weights2 = [tokfreqs2[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                    for token in tokfreqs2] if doc_freqs else None

        embedding1 = np.average([model[token] for token in tokfreqs1], axis=0, weights=weights1).reshape(1, -1)
        embedding2 = np.average([model[token] for token in tokfreqs2], axis=0, weights=weights2).reshape(1, -1)

        sim = cosine_similarity(embedding1, embedding2)[0][0]
        sims.append(sim)

    return sims

#word mover's distance
def run_wmd_benchmark(sentences1, sentences2, model, use_stoplist=False):
    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):

        tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
        tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

        tokens1 = [token for token in tokens1 if token in model]
        tokens2 = [token for token in tokens2 if token in model]

        if len(tokens1) == 0 or len(tokens2) == 0:
            tokens1 = [token for token in sent1.tokens if token in model]
            tokens2 = [token for token in sent2.tokens if token in model]

        sims.append(-model.wmdistance(tokens1, tokens2))

    return sims

from sklearn.decomposition import TruncatedSVD

def remove_first_principal_component(X):
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(X)
    pc = svd.components_
    XX = X - X.dot(pc.transpose()) * pc
    return XX


def run_sif_benchmark(sentences1, sentences2, model, freqs={}, use_stoplist=False, a=0.001):
    total_freq = sum(freqs.values())

    embeddings = []

    # SIF requires us to first collect all sentence embeddings and then perform
    # common component analysis.
    for (sent1, sent2) in zip(sentences1, sentences2):
        tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
        tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

        tokens1 = [token for token in tokens1 if token in model]
        tokens2 = [token for token in tokens2 if token in model]

        weights1 = [a / (a + freqs.get(token, 0) / total_freq) for token in tokens1]
        weights2 = [a / (a + freqs.get(token, 0) / total_freq) for token in tokens2]

        embedding1 = np.average([model[token] for token in tokens1], axis=0, weights=weights1)
        embedding2 = np.average([model[token] for token in tokens2], axis=0, weights=weights2)

        embeddings.append(embedding1)
        embeddings.append(embedding2)

    embeddings = remove_first_principal_component(np.array(embeddings))
    sims = [cosine_similarity(embeddings[idx * 2].reshape(1, -1),
                              embeddings[idx * 2 + 1].reshape(1, -1))[0][0]
            for idx in range(int(len(embeddings) / 2))]

    return sims


def run_gse_benchmark(sentences1, sentences2,embed):
    sts_input1 = tf.placeholder(tf.string, shape=(None))
    sts_input2 =  tf.placeholder(tf.string, shape=(None))

    sts_encode1 = tf.nn.l2_normalize(embed(sts_input1))
    sts_encode2 = tf.nn.l2_normalize(embed(sts_input2))

    sim_scores = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())

        [gse_sims] = session.run(
            [sim_scores],
            feed_dict={
                sts_input1: [sent1.raw for sent1 in sentences1],
                sts_input2: [sent2.raw for sent2 in sentences2]
            })
    return gse_sims


def run_experiment(df, benchmarks):
    sentences1 = [Sentence(s) for s in df['sent_1']]
    sentences2 = [Sentence(s) for s in df['sent_2']]

    pearson_cors, spearman_cors = [], []
    for label, method in benchmarks:
        sims = method(sentences1, sentences2)
        pearson_correlation = scipy.stats.pearsonr(sims, df['sim'])[0]
        print(label, pearson_correlation)
        pearson_cors.append(pearson_correlation)
        spearman_correlation = scipy.stats.spearmanr(sims, df['sim'])[0]
        spearman_cors.append(spearman_correlation)
    return pearson_cors, spearman_cors
import functools as ft
benchmarks = [("AVG-W2V", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=False)),
              ("AVG-W2V-STOP", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=True)),
              ("AVG-W2V-TFIDF", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=False, doc_freqs=doc_frequencies)),
              ("AVG-W2V-TFIDF-STOP", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=True, doc_freqs=doc_frequencies)),
              ("WMD-W2V", ft.partial(run_wmd_benchmark, model=word2vec, use_stoplist=False)),
              ("WMD-W2V-STOP", ft.partial(run_wmd_benchmark, model=word2vec, use_stoplist=True)),
              ("SIF-W2V", ft.partial(run_sif_benchmark, freqs=frequencies, model=word2vec, use_stoplist=False)),
              ("GSE", run_gse_benchmark)]
pearson_results, spearman_results = {}, {}
pearson_results["SICK-DEV"], spearman_results["SICK-DEV"] = run_experiment(sick_dev, benchmarks)
pearson_results["SICK-TEST"], spearman_results["SICK-TEST"] = run_experiment(sick_test, benchmarks)
pearson_results["STS-DEV"], spearman_results["STS-DEV"] = run_experiment(sts_dev, benchmarks)
pearson_results["STS-TEST"], spearman_results["STS-TEST"] = run_experiment(sts_test, benchmarks)