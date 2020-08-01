import pytest
import pandas as pd
import numpy as np
from faker import Faker
from sklearn.datasets import load_iris, fetch_20newsgroups

from feet import pipeline_from_config

iris_predictive_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
target_column = 'target'

def get_iris_dataframe():
    iris = load_iris()
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                      columns= iris['feature_names'] + ['target'])\
           .assign(target=lambda y: y.target.apply(lambda x: 0 if x == 0 else 1))

    return df[iris["feature_names"]], df['target']

def artificial_text(records=5000):
    fake = Faker()
    data = [{
        "text": fake.text(),
        "name": fake.name(),
        "addy": fake.address(),
        
    } for i in range(records)]


    return pd.DataFrame(data)

def fetch_20ngs():
    ngs = fetch_20newsgroups(remove=("headers", "footers", "quotes"),
                             categories=['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med'],)
    df = pd.DataFrame([{"text": d[0], "target": d[1]} for d in zip(ngs["data"], ngs["target"])]).dropna()

    return df
    

def test_text_handling():
    records = 5000
    df = artificial_text(records)

    tfidf_config = {
        "transforms": [
            {
                "type": "featurizer",
                "field": ["text"],
                "transforms": [{"name": "tfidf", "config": {"max_features": 5000}}]
            }
        ]
    }

    tokenizer_config = {
        "transforms": [
            {
                "type": "featurizer",
                "field": ["text"],
                "transforms": [{"name": "tokenizer", "config": {"max_features": 5000}}]
            }
        ]
    }

    hashing_config = {
        "transforms": [
            {
                "type": "featurizer",
                "field": ["text"],
                "transforms": [{"name": "hashing", "config": {"n_features": 5000}}]
            }
        ]
    }

    for config in [tfidf_config, tokenizer_config, hashing_config]:
        pipeline = pipeline_from_config(config)
        X = pipeline.fit_transform(df)

        assert (X.shape[0] == df.shape[0])
        assert (X.shape[0] == records)
        assert (X.shape[1] == 5000)
        assert (X.sum() > 0)


def test_lda_transform():
    records = 5000
    df = artificial_text(records)
    lda_config = {
        "transforms": [
            {
                "type": "featurizer",
                "field": ["text"],
                "transforms": [{"name": "tfidf", "config": {"max_features": 5000}}]
            }
        ],
        "post_process": [
            {"name": "lda", "config": {"rank": 50}}
        ]
    }
    lda_pipeline = pipeline_from_config(lda_config)
    X = lda_pipeline.fit_transform(df)

    assert (X.shape[0] == df.shape[0])
    assert (X.shape[1] == 50)

def test_lda_convenience_transform():
    records = 5000
    df = artificial_text(records)
    lda_config = {
        "transforms": [
            {
                "type": "featurizer",
                "field": ["text"],
                "transforms": [{"name": "lda", "config": {"rank": 50}}]
            }
        ],
    }
    lda_pipeline = pipeline_from_config(lda_config)
    X = lda_pipeline.fit_transform(df)

    assert (X.shape[0] == df.shape[0])
    assert (X.shape[1] == 50)

def test_standard_numeric_transformer():
    df, y = get_iris_dataframe()
    config = {
        "transforms": [
            {
                "type"       : "featurizer",
                "transforms" : [{"name": "standard_numeric"}],
                "field"      : iris_predictive_columns
            }
        ]
    }

    pipeline = pipeline_from_config(config)
    X = pipeline.fit_transform(df)

    assert (X.shape[1] == df.shape[1])
    assert (X.shape[0] == df.shape[0])

    for i in range(X.shape[1]):
        m = X[:, i].mean()
        s = X[:, i].std()

        assert (abs(m) < .1) # close to 0
        assert (abs(s) < 1.1) # close to 1


def test_gaussian_mixture_transformer():
    df, y = get_iris_dataframe()
    config = {
        "post_process": [
            {"name": "gmm", "config": {"clusters": 50}}
        ]
    }
    
    pipeline = pipeline_from_config(config)
    X = pipeline.fit_transform(df)

    assert (X.shape[1] == 50)
    assert (X.shape[0] == df.shape[0])

def test_kernel_pca():
    
    df, y = get_iris_dataframe()
    for kernel in ["linear", "poly", "rbf", "sigmoid", "cosine"]:
        config = {
            "post_process": [
                {"name": "kpca", "config": {"n_components": 10, "kernel": kernel}}
            ]
        }
    
        pipeline = pipeline_from_config(config)
        X = pipeline.fit_transform(df)

        assert (X.shape[1] == 10)
        assert (X.shape[0] == df.shape[0])

    
def test_handles_process_steps_individually():

    df, target = get_iris_dataframe()
    
    """ dont throw value error if a config with just a pre_process is passed"""
    pre_process_only_config = {
        "pre_process": [
            {
                "name": "fillna",
                "field": iris_predictive_columns
            }
        ],
    }

    pre_process_only_pipeline = pipeline_from_config(pre_process_only_config)

    pre_process_X = pre_process_only_pipeline.fit_transform(df)
    assert (pre_process_X.shape == df.shape)
    assert (np.isnan(pre_process_X).sum() == 0)

    """ dont throw value error if a config with just transforms are passed"""
    transform_only_config = {
        "transforms": [{"type":"featurizer", "transforms": [{"name":"quantile_numeric"}], "field": iris_predictive_columns}]
    }
    
    transform_only_pipeline = pipeline_from_config(transform_only_config)

    transform_X = transform_only_pipeline.fit_transform(df)
    assert (transform_X.shape[1] == 101*df.shape[1])

    """ dont throw value error if a config with just a post_process is passed"""
    post_process_only_config = {
        "post_process": [{"name": "svd", "config": {"rank": 2}}]
    }

    post_process_only_pipeline = pipeline_from_config(post_process_only_config)

    post_process_X = post_process_only_pipeline.fit_transform(df)
    assert (post_process_X.shape[1] == 2)

    
def test_handles_individual_pipeline_components():

    """ dont throw value error if a config with just a pre_process is passed"""
    pre_process_only_config = {
        "pre_process": [
            {
                "name": "fillna",
                "field": iris_predictive_columns,
            }
        ],
    }

    pre_process_only_pipeline = pipeline_from_config(pre_process_only_config)

    """ dont throw value error if a config with just transforms are passed"""
    transform_only_config = {
        "transforms": [{"type":"featurizer", "transforms": [{"name":"quantile_numeric"}], "field": iris_predictive_columns}]
    }
    
    transform_only_pipeline = pipeline_from_config(transform_only_config)

    """ dont throw value error if a config with just a post_process is passed"""
    post_process_only_config = {
        "post_process": [{"name": "svd", "config": {"rank": 20}}]
    }

    post_process_only_pipeline = pipeline_from_config(post_process_only_config)

def test_throws_exception_on_invalid_inputs():
    with pytest.raises(ValueError):
        config = {
            "transforms": [{"type":"featurizer", "transforms": [{"name":"quantile_numeric"}], "field":None}]
        }
        pipeline = pipeline_from_config(config)

    with pytest.raises(ValueError):
        config = {
            "transforms": [{"type":"featurizer", "transforms": [{"name":"quantile_numeric"}], "field":[]}]
        }
        pipeline = pipeline_from_config(config)

    with pytest.raises(ValueError):
        config = {
            "pre_process": [
            {
                "name": "fillna",
                "field": None
            }
            ],
            "transforms": [{"type":"featurizer", "transforms": [{"name":"quantile_numeric"}], "field": iris_predictive_columns}]
        }
        pipeline = pipeline_from_config(config)

    with pytest.raises(ValueError):
        config = {
            "pre_process": [
            {
                "name": "fillna",
                "field": []
            }
            ],
            "transforms": [{"type":"featurizer", "transforms": [{"name":"quantile_numeric"}], "field": iris_predictive_columns}]
        }
        pipeline = pipeline_from_config(config)

    with pytest.raises(ValueError):
        config = {
            "foo": {
                "bar": "baz"
            }
        }
        pipeline = pipeline_from_config(config)
