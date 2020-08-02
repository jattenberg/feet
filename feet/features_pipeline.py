import logging
import re
from collections import Counter, OrderedDict
import numpy as np
import chardet
from pandas import DataFrame
from geopandas.geodataframe import GeoDataFrame
from html2text import html2text
import os
import functools

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelBinarizer,
    StandardScaler,
    PolynomialFeatures,
    MinMaxScaler,
    Normalizer,
)
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import (
    TruncatedSVD,
    NMF,
    LatentDirichletAllocation,
    KernelPCA,
)
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
    chi2,
    f_classif,
    mutual_info_classif,
)


from flair.data import Sentence
from flair.embeddings import (
    FlairEmbeddings,
    WordEmbeddings,
    StackedEmbeddings,
    BertEmbeddings,
    ELMoEmbeddings,
    CharLMEmbeddings,
    DocumentPoolEmbeddings,
    DocumentLSTMEmbeddings,
)

import gensim


def pipeline_from_config_file(filename):
    """
        public api to access pipeline
        creation when config is a json file,
        see `pipeline_from_config` for an example
            of what that config would look like
    """
    return pipeline_from_config(json.load(open(filename, "r")))


def pipeline_from_config(config):
    """
       public api to access pipeline creation
       pipeline config defines a series of transforms
       performed to each column of the data to be processed,
       along with any post processing steps

       see for example:
config = 
    {
        pre_process: [
           {
              name: concat,
              field: [header, tags, subject],
              config: {out_field: metadata, glue: " "}
           }
        ],
        transforms: [
            {
                type: featurizer, # basic
                field: [body_text, metadata],
                transforms: [
                    {name: tfidf, config: {}}
                ]
            },
            {
                type: compound,  # recursive
                transforms: [{
                    type: featurizer,
                    field: subject_text,
                    transforms: [{name: tfidf, config: {}}]
                }],
                post_process: [{
                    name: nmf,
                    config: {}
                }]  # none is just concat
            }

         ], # there is a concat prior to post process
         post_processes: [ # a list where each element is a step. nested lists are parallel tasks
             [{name: svd, config: {}}, {name: null, config: {}}], # step 1, all done in parallel, concat'd together
             {name: norm, config: {norm: l2}}
         ]
    }

    returned pipeline can then then be instantiated with `fit`,
          and used to `transform` data to create potentially
          useful features 
    """
    if (
        not "pre_process" in config
        and not "transforms" in config
        and not "post_process" in config
    ):
        raise ValueError(
            "invalid configuration. must specify at least one of 'pre_process' 'transforms', 'post_process'"
        )
    return _compound_pipeline_from_config(config)


def _transformer_from_config(field, transformer_config):
    name = transformer_config["name"]
    configs = transformer_config.get("config", {})
    return get_transformer(name)(field, **configs)


def _transformers_from_config(field, transformers):
    transforms = [
        _transformer_from_config(field, transformer) for transformer in transformers
    ]
    return ("%s_features" % field, FeatureUnion(transformer_list=transforms))


def _handle_transform(transform):
    transform_type = transform.get("type", "featurizer")
    if transform_type == "featurizer":
        if isinstance(transform["field"], (list,)):
            if len(transform["field"]) < 1:
                raise ValueError(
                    "transforms require a non-empty list of fields to operate on"
                )
            transformer_list = [
                _transformers_from_config(field, transform["transforms"])
                for field in transform["field"]
            ]
            return (
                "%s_combined" % "_".join(transform["field"]),
                FeatureUnion(transformer_list=transformer_list),
            )
        else:
            if not transform["field"]:
                raise ValueError("Invalid or false field value")
            return _transformers_from_config(
                transform["field"], transform["transforms"]
            )
    elif transform_type == "compound":
        return ("compound", _compound_pipeline_from_config(transform))
    else:
        raise ValueError("invalid transform type: %s" % transform["type"])


def _handle_preprocess(pre_process):
    name = pre_process["name"]
    config = pre_process.get("config", {})
    field = pre_process["field"]

    if not field or (isinstance(field, (list,)) and len(field) < 1):
        raise ValueError(
            "invalid field passed for preprocessing. requires a single column name or non-empty list of column names, depending on the process"
        )

    return get_preprocess(name)(field, **config)


def _handle_postprocess(components, post_process):
    name = post_process["name"]
    configs = post_process.get("config", {})
    return get_postprocess(name)(components, **configs)


def _handle_postprocess_sequence(components, postprocess_config):
    if len(postprocess_config) > 0:
        if isinstance(postprocess_config[0], (list,)):
            logging.debug(
                "processing: %s" % ", ".join(p["name"] for p in postprocess_config[0])
            )
            processed = [
                _handle_postprocess(components, post_process)
                for post_process in postprocess_config[0]
            ]
        else:
            logging.debug(
                "processing a single post process: %s" % postprocess_config[0]["name"]
            )
            processed = [_handle_postprocess(components, postprocess_config[0])]
        out = FeatureUnion(transformer_list=processed)

        return _handle_postprocess_sequence(out, postprocess_config[1:])
    else:
        return components


def _compound_pipeline_from_config(config):
    """
       constructing complex pipelines
    """
    preprocess = (
        [_handle_preprocess(pre) for pre in config["pre_process"]]
        if "pre_process" in config
        else []
    )
    components = (
        [_handle_transform(transform) for transform in config["transforms"]]
        if "transforms" in config
        else [("identity", Pipeline([("identity", IdentityTransformer())]))]
    )

    steps = Pipeline(
        preprocess + [("components", FeatureUnion(transformer_list=components))]
    )

    if "post_process" in config and len(config["post_process"]) > 0:
        out = _handle_postprocess_sequence(steps, config["post_process"])
        return Pipeline([("post_processed", out)])
    else:
        return steps


def get_transformer(name):
    """
       some convenience methods for common
       feature creation methods

       todo: handle more manual feature creation pipelines
    """
    transformer_map = {
        "standard": build_numeric_column,
        "standard_numeric": build_numeric_column,
        "quantile_numeric": build_quantile_column,
        "range_numeric": build_range_scaler,
        "dummyizer": build_dummyizer,
        "null_transformer": build_null,  # don't do anything to this column
        "tokenizer": build_word_tokenizer,
        "array_vocab": build_array_vocabizer,
        "tfidf": build_tfidf_transformer,
        "word_count": build_wordcount_transformer,
        "hashing": build_feature_hashing_transformer,
        "w2v": build_word2vec_transformer,
        "gs_lda": build_gs_lda_transformer,
        "flair": build_flair_transformer,
        "lda": build_lda_shortcut,
    }
    return transformer_map[name]


def get_preprocess(name):
    processor_map = {
        "concat": build_field_concatter,
        "fillna": build_na_filler,
    }

    return processor_map[name]


def get_postprocess(name):
    processor_map = {
        "null": build_null_pipeline,
        "nmf": build_nmf,
        "svd": build_svd,
        "lda": build_lda,
        "rte": build_rte,
        "poly": build_polynomial,
        "abs": build_abs,
        "norm": build_norm,
        "std": build_standardizer,
        "select": build_feature_selector,
        "kmeans": build_kmeans_embedder,
        "gmm": build_gmm_embedder,
        "kpca": build_kernel_pca_embedder,
    }

    return processor_map[name]


"""
   #####################################################################################
   featurizer convenience methods
   #####################################################################################
"""


def build_numeric_column(col):
    return (
        "numeric_%s" % col,
        Pipeline(
            [
                ("selector", ItemSelector(col)),
                ("reshaper", Reshaper()),
                ("floater", Floater()),
                ("scaler", StandardScaler()),
            ]
        ),
    )


def build_quantile_column(col, n_quantiles=100):
    return (
        "quantile_%s" % col,
        Pipeline(
            [
                ("selector", ItemSelector(col)),
                ("reshaper", Reshaper()),
                ("quantiler", Quantiler(n_quantiles)),
            ]
        ),
    )


def build_range_scaler(col, min=0, max=1):
    return (
        "min_max %s" % col,
        Pipeline(
            [
                ("selector", ItemSelector(col)),
                ("reshaper", Reshaper()),
                ("min_max", MinMaxScaler(feature_range=(min, max))),
            ]
        ),
    )


def build_dummyizer(col):
    return (
        "onehot_s_%s" % col,
        Pipeline([("selector", ItemSelector(col)), ("label", Dummyizer()),]),
    )


def build_null(col):
    return (
        "null_%s" % col,
        Pipeline([("selector", ItemSelector(col)), ("reshaper", Reshaper())]),
    )


def build_feature_hashing_transformer(col, n_features=2 ** 20, alternate_sign=True):

    return (
        "hasher_%s" % col,
        Pipeline(
            [
                ("selector", ItemSelector(col)),
                ("cleaner", WordCleaner()),
                (
                    "hasher",
                    FeatureHasher(
                        input_type="string",
                        n_features=n_features,
                        alternate_sign=alternate_sign,
                    ),
                ),
            ]
        ),
    )


def build_wordcount_transformer(col, binary=False, min_df=0.0, ngrams=2):
    return (
        "wordcount_%s" % col,
        Pipeline(
            [
                ("selector", ItemSelector(col)),
                ("cleaner", WordCleaner()),
                (
                    "counter",
                    CountVectorizer(
                        binary=binary,
                        min_df=min_df,
                        decode_error="ignore",
                        ngram_range=(1, ngrams),
                    ),
                ),
            ]
        ),
    )


def identity(x):
    return x


def build_word_tokenizer(
    col, binary=False, min_df=0.0, max_df=1.0, max_features=None, seperator=","
):
    return (
        "word_tokens_%s" % col,
        Pipeline(
            [
                ("selector", ItemSelector(col)),
                ("splitter", WordSplitter(splitter=seperator)),
                (
                    "counter",
                    TfidfVectorizer(
                        input="content",
                        binary=binary,
                        use_idf=False,
                        min_df=min_df,
                        max_df=max_df,
                        max_features=max_features,
                        tokenizer=identity,
                        preprocessor=identity,
                    ),
                ),
            ]
        ),
    )


def build_array_vocabizer(col, binary=False, min_df=0.0, max_df=1.0, max_features=None):
    """
        the specified column is an array of tokens to be used as a feature
    """
    return (
        "array_vocabizer_%s" % col,
        Pipeline(
            [
                ("selector", ItemSelector(col)),
                (
                    "counter",
                    TfidfVectorizer(
                        input="content",
                        binary=binary,
                        use_idf=False,
                        min_df=min_df,
                        max_df=max_df,
                        max_features=max_features,
                        tokenizer=identity,
                        preprocessor=identity,
                    ),
                ),
            ]
        ),
    )


def build_tfidf_transformer(col, min_df=0.0, max_df=1.0, max_features=None, ngrams=2):
    return (
        "tfidf_%s" % col,
        Pipeline(
            [
                ("selector", ItemSelector(col)),
                ("cleaner", WordCleaner()),
                (
                    "tfidf",
                    TfidfVectorizer(
                        min_df=min_df,
                        max_df=max_df,
                        max_features=max_features,
                        decode_error="ignore",
                        ngram_range=(1, ngrams),
                    ),
                ),
            ]
        ),
    )


def build_gs_lda_transformer(col, replace_pattern=r"[^0-9a-zA-Z ]", topics=50):
    return (
        "gs_lda_%s" % col,
        Pipeline(
            [
                ("selector", ItemSelector(col)),
                ("cleaner", WordCleaner()),
                ("replacer", WordReplacer(replace_pattern=replace_pattern)),
                ("splitter", WordSplitter()),
                ("w2v", LDAifier(topics=topics)),
            ]
        ),
    )


def build_flair_transformer(
    col,
    replace_pattern=r"[^0-9a-zA-Z ]",
    embeddings=["flair-forward"],
    pooling="doc-pooling",
):
    return (
        "flair_embed_%s" % col,
        Pipeline(
            [
                ("selector", ItemSelector(col)),
                ("cleaner", WordCleaner()),
                ("replacer", WordReplacer(replace_pattern=replace_pattern)),
                (
                    "flaired",
                    FlairEmbeddingGenerator(embeddings=embeddings, pooling=pooling),
                ),
            ]
        ),
    )


def build_lda_shortcut(
    col, rank=50, min_df=0.0, max_df=1.0, max_features=None, ngrams=2
):

    return (
        "lda_%s" % col,
        Pipeline(
            [
                ("selector", ItemSelector(col)),
                ("cleaner", WordCleaner()),
                (
                    "tfidf",
                    TfidfVectorizer(
                        min_df=min_df,
                        max_df=max_df,
                        max_features=max_features,
                        decode_error="ignore",
                        ngram_range=(1, ngrams),
                    ),
                ),
                ("lda", LatentDirichletAllocation(n_components=rank, n_jobs=-1)),
            ]
        ),
    )


def build_word2vec_transformer(
    col,
    replace_pattern=r"[^0-9a-zA-Z ]",
    rank=100,
    window=50,
    workers=1,
    alpha=0.25,
    min_count=5,
    max_vocab_size=None,
    negative=5,
    cbow_mean=1,
    skip_gram=False,
):
    return (
        "w2v_%s" % col,
        Pipeline(
            [
                ("selector", ItemSelector(col)),
                ("cleaner", WordCleaner()),
                ("replacer", WordReplacer(replace_pattern=replace_pattern)),
                ("splitter", WordSplitter()),
                (
                    "w2v",
                    W2Vifier(
                        rank=rank,
                        window=window,
                        workers=workers,
                        alpha=alpha,
                        min_count=min_count,
                        max_vocab_size=max_vocab_size,
                        negative=negative,
                        cbow_mean=cbow_mean,
                        skip_gram=skip_gram,
                    ),
                ),
            ]
        ),
    )


"""
   #####################################################################################
   pre processor convenience methods
   #####################################################################################
"""


def build_field_concatter(cols, out_field, glue=" "):
    return (
        "concatter_%s" % "_".join(cols),
        Pipeline([("concat_cols", Concatenator(cols, out_field=out_field, glue=glue))]),
    )


def build_na_filler(col, value=0):
    return (
        "na_filler_%s" % col,
        Pipeline([("na_filler", NAFiller(col=col, value=value))]),
    )


"""
   #####################################################################################
   post processor convenience methods
   #####################################################################################
"""


def build_null_pipeline(pipeline):
    return ("null_pipeline", pipeline)


def build_polynomial(pipeline, degree=2, interaction_only=False, include_bias=True):
    return (
        "polynomial",
        Pipeline(
            [
                ("preprocessed", pipeline),
                ("densinator", Densinator()),
                (
                    "poly",
                    PolynomialFeatures(
                        degree=degree,
                        interaction_only=interaction_only,
                        include_bias=include_bias,
                    ),
                ),
            ]
        ),
    )


def build_svd(pipeline, rank=50):
    return (
        "svd",
        Pipeline(
            [("preprocessed", pipeline), ("svd", TruncatedSVD(n_components=rank))]
        ),
    )


def build_kmeans_embedder(pipeline, clusters=10):
    return (
        "kmeans",
        Pipeline([("preprocessed", pipeline), ("kmeans", KMeans(n_clusters=clusters))]),
    )


def build_gmm_embedder(pipeline, clusters=10, covariance="full"):
    return (
        "gmm",
        Pipeline(
            [
                ("preprocessed", pipeline),
                (
                    "gmm",
                    GaussianMixtureModel(n_components=clusters, covariance=covariance),
                ),
            ]
        ),
    )


def build_kernel_pca_embedder(
    pipeline, n_components=50, kernel="linear", degree=3, gamma=None
):

    return (
        "pca",
        Pipeline(
            [
                ("preprocessed", pipeline),
                (
                    "gmm",
                    KernelPCA(
                        n_components=n_components,
                        kernel=kernel,
                        degree=degree,
                        gamma=gamma,
                    ),
                ),
            ]
        ),
    )


def build_rte(
    pipeline,
    n_estimators=100,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    min_impurity_decrease=0.000001,
):

    return (
        "rte",
        Pipeline(
            [
                ("preprocessed", pipeline),
                (
                    "rte",
                    RandomTreesEmbedding(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        min_impurity_decrease=min_impurity_decrease,
                    ),
                ),
            ]
        ),
    )


def build_nmf(pipeline, rank=50):

    return (
        "nmf",
        Pipeline([("preprocessed", pipeline), ("nmf", NMF(n_components=rank))]),
    )


def build_lda(pipeline, rank=50):

    return (
        "lda",
        Pipeline(
            [
                ("preprocessed", pipeline),
                ("lda", LatentDirichletAllocation(n_components=rank, n_jobs=-1)),
            ]
        ),
    )


def build_abs(pipeline):
    return ("abs", Pipeline([("preprocessed", pipeline), ("abs", AbsoluteValue())]))


def build_norm(pipeline, norm="l2"):
    return (
        "normalizer",
        Pipeline([("preprocessed", pipeline), ("normalizer", Normalizer(norm))]),
    )


def build_feature_selector(pipeline, to_get, how="f_classif"):
    selectors = {
        "f_classif": f_classif,
        "chi2": chi2,
        "mutual_info_classif": mutual_info_classif,
        "f_regression": f_regression,
        "mutual_info_regression": mutual_info_regression,
    }

    return (
        "feature_selector",
        Pipeline(
            [
                ("preprocessed", pipeline),
                ("selector", SelectKBest(k=to_get, how=selectors[how])),
            ]
        ),
    )


def build_standardizer(pipeline):
    return (
        "standardizer",
        Pipeline([("preprocessed", pipeline), ("standardized", StandardScaler())]),
    )


def build_recursive_postprocess(pipeline, post_process_list):
    if len(post_process_list) > 0:
        process = post_process_list[0]
        processed = Pipeline([_handle_postprocess(pipeline, process)])
        return build_recursive_postprocess(processed, post_process_list[1:])
    else:
        return ("recursed", pipeline)


"""
   #####################################################################################
   custom pipeline components
   #####################################################################################
"""


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if type(X) in {DataFrame, GeoDataFrame}:
            return X[self.key]
        else:
            raise NotImplementedError(
                "unsupported itemselector type. implement some new stuff: %s" % type(X)
            )


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    does nothing
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class Reshaper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, None]


class Dummyizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.dummyizer = LabelBinarizer(sparse_output=True)
        self.dummyizer.fit(X)
        return self

    def transform(self, X):
        return self.dummyizer.transform(X)


class NAFiller(BaseEstimator, TransformerMixin):
    def __init__(self, col, value=0):
        self.col = col
        self.value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def _filler(c):
            def _inner_filler(df):
                return df[c].fillna(value=self.value)

            return (c, _inner_filler)

        if isinstance(self.col, (list,)):
            trans = dict([_filler(c) for c in self.col])
            return X.assign(**trans)
        else:
            return X.assign(**{_filler(self.col)})


class Concatenator(BaseEstimator, TransformerMixin):
    def __init__(self, cols, out_field, glue=" "):
        self.cols = cols
        self.out_field = out_field
        self.glue = glue

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def _inner_joiner(row):
            return self.glue.join(row.values.astype(str))

        def _joiner(df):
            return df[self.cols].apply(_inner_joiner, axis=1)

        out = X.assign(**{self.out_field: _joiner})
        return out


class Floater(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype("float64")


class Densinator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.todense()


class Quantiler(BaseEstimator, TransformerMixin):
    def __init__(self, n_quantiles=100):
        self.n_quantiles = n_quantiles

    def fit(self, X, y=None):
        percentiles = np.linspace(0, 100, self.n_quantiles + 2)
        self.quantiles = np.percentile(X, percentiles)
        return self

    def find_quantile(self, x):
        return [
            1 if self.quantiles[i] < x and self.quantiles[i + 1] >= x else 0
            for i in range(0, len(self.quantiles) - 1)
        ]

    def transform(self, X):
        return [self.find_quantile(x) for x in X]


class WordReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, replace_pattern=r"[^0-9a-zA-Z ]", replacement=""):
        self.replace_pattern = replace_pattern
        self.replacement = replacement

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [re.sub(self.replace_pattern, self.replacement, x) for x in X]


class WordSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, splitter="\s+"):
        self.splitter = splitter

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [[y.strip() for y in re.split(self.splitter, x) if y.strip()] for x in X]


class LDAifier(BaseEstimator, TransformerMixin):
    def __init__(self, topics=50, passes=50):
        self.topics = topics
        self.passes = passes
        self.lda = None
        self.dictionary = None

    def fit(self, X, y=None):
        self.dictionary = gensim.corpora.Dictionary(X)

        term_doc_matrix = [self.dictionary.doc2bow(doc) for doc in X]
        logging.info("fitting a gensim lda model")
        self.lda = gensim.models.ldamulticore.LdaMulticore(
            term_doc_matrix,
            num_topics=self.topics,
            passes=self.passes,
            id2word=self.dictionary,
            workers=None,
        )
        logging.info("done")

        return self

    def embed(self, document):
        # list of (int, float)
        topic_probas = self.lda.get_document_topics(self.dictionary.doc2bow(document))

        out = np.zeros(self.topics)
        # is there a functional way to do this?
        for t, p in topic_probas:
            out[t] = p

        return out

    def transform(self, X):
        return [self.embed(x) for x in X]


class W2Vifier(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rank=100,
        window=50,
        workers=1,
        alpha=0.25,
        min_count=5,
        max_vocab_size=None,
        negative=5,
        cbow_mean=1,
        skip_gram=False,
    ):
        self.rank = rank
        self.window = window
        self.workers = workers
        self.alpha = alpha
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.negative = negative
        self.cbow_mean = cbow_mean
        self.skip_gram = skip_gram
        self.w2v = None

    def fit(self, X, y=None):
        logging.info("fitting w2v model")
        self.w2v = gensim.models.Word2Vec(
            X,
            size=self.rank,
            window=self.window,
            workers=self.workers,
            alpha=self.alpha,
            min_count=self.min_count,
            max_vocab_size=self.max_vocab_size,
            negative=self.negative,
            cbow_mean=self.cbow_mean,
            sg=1 if self.skip_gram else 0,
        )
        logging.info("done fitting w2v")
        return self

    def embed(self, x):
        def add(x, y):
            return x + y

        out = functools.reduce(
            add,
            [self.w2v.wv[word] for word in x if word in self.w2v.wv],
            np.zeros(self.rank),
        )
        norm = np.linalg.norm(out)
        if norm > 0:
            return out / norm
        else:
            # vector of zeros (no vocab in model)
            return out

    def transform(self, X):
        return [self.embed(x) for x in X]


class FlairEmbeddingGenerator(BaseEstimator, TransformerMixin):
    def _collect_embeddings(self, embeddings):
        embedders = {
            "flair-forward": FlairEmbeddings("multi-forward"),
            "flair-backward": FlairEmbeddings("multi-backward"),
            "charm-forward": CharLMEmbeddings("news-forward"),
            "charm-backward": CharLMEmbeddings("news-backward"),
            "glove": WordEmbeddings("glove"),
            "bert-small": BertEmbeddings("bert-base-uncased"),
            "bert-large": BertEmbeddings("bert-large-uncased"),
            "elmo-small": ELMoEmbeddings("small"),
            "elmo-large": ELMoEmbeddings("original"),
        }

        return [embedders[embedding] for embedding in embeddings]

    def __init__(self, embeddings=["flair-forward"], pooling="doc-pooling"):
        self.embeddings = embeddings
        self.pooling = pooling

        poolers = {
            "lstm": DocumentLSTMEmbeddings,
            "doc-pooling": DocumentPoolEmbeddings,
        }
        self.embedder = poolers[self.pooling](self._collect_embeddings(self.embeddings))

    def fit(self, X, y=None):
        return self

    def embed_sentence(self, x):
        sentence = Sentence(x)
        self.embedder.embed(sentence)
        return sentence.get_embedding().numpy().squeeze()

    def transform(self, X):
        return [self.embed_sentence(x) for x in X]


class WordCleaner(BaseEstimator, TransformerMixin):
    def decode(self, content):
        str_bytes = str.encode(content)
        charset = chardet.detect(str_bytes)["encoding"]
        return str_bytes.decode(encoding=charset, errors="ignore")

    feature_regex_pipe = [
        (r"\|", " "),
        (r"\r\n?|\n", " "),
        (r"[^\x00-\x7F]+", " "),
        (r"\s+", " "),
        (r"https?://\S+", "_url_"),
        (r"\w{,20}[a-zA-Z]{1,20}[0-9]{1,20}", "_wn_"),
        (r"\d+/\d+/\d+", "_d2_"),
        (r"\d+/\d+", "_d_"),
        (r"\d+:\d+:\d+", "_ts_"),
        (r":", " "),
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def _text_clean(x):
            def apply_regex(acc, re_rep):
                return re.sub(re_rep[0], re_rep[1], acc)

            all_clean = html2text(self.decode(x))
            replaced = functools.reduce(apply_regex, self.feature_regex_pipe, all_clean)
            return " ".join([y for y in replaced.split(" ") if len(y) <= 20])

        return map(_text_clean, X)


class AbsoluteValue(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [abs(x) for x in X]


class GaussianMixtureModel(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10, covariance="full"):
        self.gmm = GaussianMixture(
            n_components=n_components, covariance_type=covariance
        )

    def fit(self, X, y=None):
        self.gmm.fit(X)
        return self

    def transform(self, X, y=None):
        return self.gmm.predict_proba(X)
