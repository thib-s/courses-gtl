# data managment
import timeit

import pandas as pd
import numpy as np
import scipy as sp
from sklearn.externals import joblib
import os.path

# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# metrics
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.model_selection import GridSearchCV

# learning
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from GaussianMixtureTransform import GaussianMixtureTransform

#######################################
# ipyparallel
#######################################
from sklearn.externals.joblib import Parallel, parallel_backend, register_parallel_backend
import ipyparallel as ipp
from ipyparallel import Client
from ipyparallel.joblib import IPythonParallelBackend

c = ipp.Client()
print(c.ids)
bview = c.load_balanced_view()

# this is taken from the ipyparallel source code
register_parallel_backend('ipyparallel', lambda: IPythonParallelBackend(view=bview))

##################################
# load datasets
##################################
fashion_df = pd.read_csv("datasets/fashion_mnist/fashion-mnist_train.csv")
fashion_df_test = pd.read_csv("datasets/fashion_mnist/fashion-mnist_test.csv")

(n_row, n_col) = fashion_df.values.shape
X_fashion = fashion_df.values[:, 1:n_col]
y_fashion = fashion_df.values[:, 0]
X_fashion_test = fashion_df_test.values[:, 1:n_col]
y_fashion_test = fashion_df_test.values[:, 0]

###################################
# KMEANS
###################################


if os.path.isfile('gridsearch_kmeans_fashion.pkl'):
    clf = joblib.load('gridsearch_kmeans_fashion.pkl')
else:
    with parallel_backend('ipyparallel'):
        param_grid = {'n_clusters': range(1, 100)}
        kmeans = KMeans()
        clf = GridSearchCV(kmeans, param_grid, cv=2, verbose=1, n_jobs=-1, pre_dispatch=7)
        clf.fit(X_fashion)
        joblib.dump(clf, 'gridsearch_kmeans_fashion.pkl')

###################################
# EM
###################################
if os.path.isfile('gridsearch_em_fashion.pkl'):
    clf = joblib.load('gridsearch_em_fashion.pkl')
else:
    with parallel_backend('ipyparallel'):
        param_grid = {'n_components': range(1, 100)}
        gm = GaussianMixture()
        clf = GridSearchCV(gm, param_grid, cv=2, verbose=1, n_jobs=-1, pre_dispatch=7)
        clf.fit(X_fashion)
        joblib.dump(clf, 'gridsearch_em_fashion.pkl')

if os.path.isfile('em_16_fashion.pkl'):
    clf = joblib.load('em_16_fashion.pkl')
else:
    with parallel_backend('ipyparallel'):
        gm = GaussianMixture(n_components=16)
        gm.fit(X_fashion, y_fashion)
        joblib.dump(clf, 'em_16_fashion.pkl')

if os.path.isfile('fashion_s.pkl'):
    clf = joblib.load('fashion_s.pkl')
else:
    with parallel_backend('ipyparallel'):
        s = sp.linalg.svd(X_fashion, full_matrices=False, compute_uv=False)
        joblib.dump(s, 'fashion_s.pkl')

#######################################
# NN tests
#######################################

nn = MLPClassifier(solver='adam', alpha=1e-5, max_iter=80000, hidden_layer_sizes=(50, 50, 50), random_state=1)
if os.path.isfile('nn_fashion.pkl'):
    nn = joblib.load('nn_fashion.pkl')
else:
    with parallel_backend('ipyparallel'):
        nn.fit(X_fashion, y_fashion)
        print(nn.score(X_fashion_test, y_fashion_test))
        joblib.dump(clf, 'nn_fashion.pkl')

if os.path.isfile('grid_nn_fashion.pkl'):
    clf = joblib.load('grid_nn_fashion.pkl')
else:
    # with parallel_backend('ipyparallel'):
    pipeline_fashion = Pipeline(
        [
            ('ica', FastICA(algorithm='parallel')),
            ('nn',
             MLPClassifier(solver='adam', alpha=1e-5, max_iter=80000, hidden_layer_sizes=(50, 50, 50), random_state=1))
        ])
    param_grid = {'ica__n_components': range(1, 100, 10)}
    clf = GridSearchCV(pipeline_fashion, param_grid, cv=2, verbose=1, n_jobs=-1)  # , pre_dispatch=14)
    clf.fit(X_fashion, y_fashion)  # , n_jobs=4, pre_dispatch=1)
    joblib.dump(clf, 'grid_nn_fashion.pkl')

if os.path.isfile('pca_nn_fashion.pkl'):
    clf = joblib.load('pca_nn_fashion.pkl')
else:
    # with parallel_backend('ipyparallel'):
    pipeline_fashion = Pipeline(
        [
            ('pca', PCA(svd_solver='full')),
            ('nn',
             MLPClassifier(solver='adam', alpha=1e-5, max_iter=80000, hidden_layer_sizes=(50, 50, 50), random_state=1))
        ]
    )
    param_grid = {'pca__n_components': range(1, 100, 10)}
    clf = GridSearchCV(pipeline_fashion, param_grid, cv=2, verbose=1, n_jobs=5, pre_dispatch=5)
    clf.fit(X_fashion, y_fashion)  # , n_jobs=4, pre_dispatch=1)
    joblib.dump(clf, 'pca_nn_fashion.pkl')

if os.path.isfile('rp_nn_fashion.pkl'):
    clf = joblib.load('rp_nn_fashion.pkl')
else:
    # with parallel_backend('ipyparallel'):
    pipeline_fashion = Pipeline(
        [
            ('rp', GaussianRandomProjection()),
            ('nn',
             MLPClassifier(solver='adam', alpha=1e-5, max_iter=80000, hidden_layer_sizes=(50, 50, 50), random_state=1))
        ]
    )
    param_grid = {'rp__n_components': range(1, 100, 10)}
    clf = GridSearchCV(pipeline_fashion, param_grid, cv=2, verbose=1, n_jobs=5, pre_dispatch=5)
    clf.fit(X_fashion, y_fashion)  # , n_jobs=4, pre_dispatch=1)
    joblib.dump(clf, 'rp_nn_fashion.pkl')

if os.path.isfile('km_nn_fashion.pkl'):
    clf = joblib.load('km_nn_fashion.pkl')
else:
    # with parallel_backend('ipyparallel'):
    pipeline_fashion = Pipeline(
        [
            ('km', KMeans()),
            ('nn',
             MLPClassifier(solver='adam', alpha=1e-5, max_iter=80000, hidden_layer_sizes=(50, 50, 50), random_state=1))
        ]
    )
    param_grid = {'km__n_clusters': range(1, 100, 10)}
    clf = GridSearchCV(pipeline_fashion, param_grid, cv=2, verbose=1, n_jobs=5, pre_dispatch=5)
    clf.fit(X_fashion, y_fashion)  # , n_jobs=16, pre_dispatch=16)
    joblib.dump(clf, 'km_nn_fashion.pkl')

if os.path.isfile('em_nn_fashion.pkl'):
    clf = joblib.load('em_nn_fashion.pkl')
else:
    with parallel_backend('ipyparallel'):
        pipeline_fashion = Pipeline(
            [
                ('em', GaussianMixtureTransform()),
                ('nn', MLPClassifier(solver='adam', alpha=1e-5, max_iter=80000, hidden_layer_sizes=(50, 50, 50),
                                     random_state=1))
            ]
        )
        param_grid = {'em__n_components': range(1, 100, 10)}
        clf = GridSearchCV(pipeline_fashion, param_grid, cv=2, verbose=1, n_jobs=5, pre_dispatch=5)
        clf.fit(X_fashion, y_fashion)  # , n_jobs=4, pre_dispatch=1)
        joblib.dump(clf, 'em_nn_fashion.pkl')

#########################################
# REDUX + CLUST
#########################################

if os.path.exists("dumps/grid_fashion_redux_cluster_em_score.pkl"):
    pass
else:
    scores = np.zeros((15, 3))
    times = np.zeros((15, 3))
    for i in range(15):
        pca = Pipeline([
            ('pca', PCA(n_components=(i * 10 + 5))),
            ('gm', GaussianMixture(n_components=10, covariance_type='diag'))
        ])
        begin = timeit.default_timer()
        pca.fit(X_fashion, y_fashion)
        times[i, 0] = timeit.default_timer() - begin
        scores[i, 0] = normalized_mutual_info_score(pca.predict(X_fashion), y_fashion)
        ica = Pipeline([
            ('ica', FastICA(n_components=(i * 10 + 5))),
            ('gm', GaussianMixture(n_components=10, covariance_type='diag'))
        ])
        begin = timeit.default_timer()
        ica.fit(X_fashion, y_fashion)
        times[i, 1] = timeit.default_timer() - begin
        scores[i, 1] = normalized_mutual_info_score(ica.predict(X_fashion), y_fashion)
        rca = Pipeline([
            ('rca', GaussianRandomProjection(n_components=(i * 10 + 5))),
            ('gm', GaussianMixture(n_components=10, covariance_type='diag'))
        ])
        begin = timeit.default_timer()
        rca.fit(X_fashion, y_fashion)
        times[i, 2] = timeit.default_timer() - begin
        scores[i, 2] = normalized_mutual_info_score(rca.predict(X_fashion), y_fashion)
        print("it:" + str(i) + "done")
    scores.dump("dumps/grid_fashion_redux_cluster_em_score.pkl")
    times.dump("dumps/grid_fashion_redux_cluster_em_times.pkl")

if os.path.exists("dumps/grid_fashion_redux_cluster_score.pkl"):
    pass
else:
    scores = np.zeros((15, 3))
    times = np.zeros((15, 3))
    for i in range(15):
        pca = Pipeline([
            ('pca', PCA(n_components=(i * 10 + 5))),
            ('kmeans', MiniBatchKMeans(n_clusters=10))
        ])
        begin = timeit.default_timer()
        pca.fit(X_fashion, y_fashion)
        times[i, 0] = timeit.default_timer() - begin
        scores[i, 0] = normalized_mutual_info_score(pca.predict(X_fashion), y_fashion)
        ica = Pipeline([
            ('ica', FastICA(n_components=(i * 10 + 5))),
            ('kmeans', MiniBatchKMeans(n_clusters=10))
        ])
        begin = timeit.default_timer()
        ica.fit(X_fashion, y_fashion)
        times[i, 1] = timeit.default_timer() - begin
        scores[i, 1] = normalized_mutual_info_score(ica.predict(X_fashion), y_fashion)
        rca = Pipeline([
            ('rca', GaussianRandomProjection(n_components=(i * 10 + 5))),
            ('kmeans', MiniBatchKMeans(n_clusters=10))
        ])
        begin = timeit.default_timer()
        rca.fit(X_fashion, y_fashion)
        times[i, 2] = timeit.default_timer() - begin
        scores[i, 2] = normalized_mutual_info_score(rca.predict(X_fashion), y_fashion)
        print("it:" + str(i) + "done")
    scores.dump("dumps/grid_fashion_redux_cluster_score.pkl")
    times.dump("dumps/grid_fashion_redux_cluster_times.pkl")
