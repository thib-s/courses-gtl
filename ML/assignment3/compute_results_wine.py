# data managment
import pandas as pd
import scipy as sp
from sklearn.externals import joblib
import os.path

# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# metrics
from sklearn.model_selection import GridSearchCV

# learning
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from GaussianMixtureTransform import GaussianMixtureTransform

#######################################
# ipyparallel
#######################################
# from sklearn.externals.joblib import Parallel, parallel_backend, register_parallel_backend
# import ipyparallel as ipp
# from ipyparallel import Client
# from ipyparallel.joblib import IPythonParallelBackend
# 
# c = ipp.Client()
# print(c.ids)
# bview = c.load_balanced_view()
# 
# # this is taken from the ipyparallel source code
# register_parallel_backend('ipyparallel', lambda: IPythonParallelBackend(view=bview))

##################################
# load datasets
##################################
winemag_df = pd.read_csv("datasets/wine_reviews/winemag-data-130k-v2.csv")
X_winemag = winemag_df['description']
y_wine = winemag_df['points']
pipeline_wine = TfidfVectorizer(use_idf=True)
pipeline_wine.fit(X_winemag, y_wine)
X_wine = pipeline_wine.transform(X_winemag)

###################################
# KMEANS
###################################


if os.path.isfile('gridsearch_kmeans_wine.pkl'):
    clf = joblib.load('gridsearch_kmeans_wine.pkl')
else:
    #with parallel_backend('ipyparallel'):
    param_grid = {'n_clusters': range(1, 100)}
    kmeans = KMeans()
    clf = GridSearchCV(kmeans, param_grid, cv=2, verbose=1, n_jobs=-1, pre_dispatch=7)
    clf.fit(X_wine)
    joblib.dump(clf, 'gridsearch_kmeans_wine.pkl')

###################################
# EM
###################################
if os.path.isfile('gridsearch_em_wine.pkl'):
    clf = joblib.load('gridsearch_em_wine.pkl')
else:
    #with parallel_backend('ipyparallel'):
    param_grid = {'n_components': range(1, 100)}
    gm = GaussianMixture()
    clf = GridSearchCV(gm, param_grid, cv=2, verbose=1, n_jobs=-1, pre_dispatch=7)
    clf.fit(X_wine)
    joblib.dump(clf, 'gridsearch_em_wine.pkl')

if os.path.isfile('em_16_wine.pkl'):
    clf = joblib.load('em_16_wine.pkl')
else:
    #with parallel_backend('ipyparallel'):
    gm = GaussianMixture(n_components=16)
    gm.fit(X_wine, y_wine)
    joblib.dump(clf, 'em_16_wine.pkl')

if os.path.isfile('wine_s.pkl'):
    clf = joblib.load('wine_s.pkl')
else:
    #with parallel_backend('ipyparallel'):
    s = sp.linalg.svd(X_wine, full_matrices=False, compute_uv=False)
    joblib.dump(s, 'wine_s.pkl')

#######################################
# ICA
#######################################
if os.path.exists('ICA_3D_winemag.pkl'):
    joblib.load('ICA_3D_winemag.pkl')
else:
    X_winemag_small = winemag_df['description'].values[0:5000]
    y_winemag_small = winemag_df['points'].values[0:5000]
    pipeline_wine_small = TfidfVectorizer()
    pipeline_wine_small.fit(X_winemag_small, y_winemag_small)
    X_w_small = pipeline_wine.transform(X_winemag_small).toarray()
    ica = FastICA(n_components=36)
    ica.fit(X_w_small)
    joblib.dump(ica, 'ICA_3D_winemag.pkl')


if os.path.exists('ICA_36D_winemag.pkl'):
    joblib.load('ICA_36D_winemag.pkl')
else:
    X_winemag_small = winemag_df['description'].values[0:5000]
    y_winemag_small = winemag_df['points'].values[0:5000]
    pipeline_wine_small = TfidfVectorizer()
    pipeline_wine_small.fit(X_winemag_small, y_winemag_small)
    X_w_small = pipeline_wine.transform(X_winemag_small).toarray()
    ica = FastICA(n_components=36)
    ica.fit(X_w_small)
    joblib.dump(ica, 'ICA_36D_winemag.pkl')

#######################################
# NN tests
#######################################

nn = MLPClassifier(solver='adam', alpha=1e-5, max_iter=80000, hidden_layer_sizes=(50, 50, 50), random_state=1)
if os.path.isfile('nn_wine.pkl'):
    nn = joblib.load('nn_wine.pkl')
else:
    #with parallel_backend('ipyparallel'):
    nn.fit(X_wine, y_wine)
    joblib.dump(clf, 'nn_wine.pkl')

if os.path.isfile('grid_nn_wine.pkl'):
    clf = joblib.load('grid_nn_wine.pkl')
else:
    # #with parallel_backend('ipyparallel'):
    pipeline_wine = Pipeline(
        [
            ('ica', FastICA(algorithm='parallel')),
            ('nn',
             MLPClassifier(solver='adam', alpha=1e-5, max_iter=80000, hidden_layer_sizes=(50, 50, 50), random_state=1))
        ])
    param_grid = {'ica__n_components': range(1, 100, 10)}
    clf = GridSearchCV(pipeline_wine, param_grid, cv=2, verbose=1, n_jobs=-1)  # , pre_dispatch=14)
    clf.fit(X_wine, y_wine)  # , n_jobs=4, pre_dispatch=1)
    joblib.dump(clf, 'grid_nn_wine.pkl')

if os.path.isfile('pca_nn_wine.pkl'):
    clf = joblib.load('pca_nn_wine.pkl')
else:
    # #with parallel_backend('ipyparallel'):
    pipeline_wine = Pipeline(
        [
            ('pca', PCA(svd_solver='full')),
            ('nn',
             MLPClassifier(solver='adam', alpha=1e-5, max_iter=80000, hidden_layer_sizes=(50, 50, 50), random_state=1))
        ]
    )
    param_grid = {'pca__n_components': range(1, 100, 10)}
    clf = GridSearchCV(pipeline_wine, param_grid, cv=2, verbose=1, n_jobs=5, pre_dispatch=5)
    clf.fit(X_wine, y_wine)  # , n_jobs=4, pre_dispatch=1)
    joblib.dump(clf, 'pca_nn_wine.pkl')

if os.path.isfile('rp_nn_wine.pkl'):
    clf = joblib.load('rp_nn_wine.pkl')
else:
    # #with parallel_backend('ipyparallel'):
    pipeline_wine = Pipeline(
        [
            ('rp', GaussianRandomProjection()),
            ('nn',
             MLPClassifier(solver='adam', alpha=1e-5, max_iter=80000, hidden_layer_sizes=(50, 50, 50), random_state=1))
        ]
    )
    param_grid = {'rp__n_components': range(1, 100, 10)}
    clf = GridSearchCV(pipeline_wine, param_grid, cv=2, verbose=1, n_jobs=5, pre_dispatch=5)
    clf.fit(X_wine, y_wine)  # , n_jobs=4, pre_dispatch=1)
    joblib.dump(clf, 'rp_nn_wine.pkl')

if os.path.isfile('km_nn_wine.pkl'):
    clf = joblib.load('km_nn_wine.pkl')
else:
    # #with parallel_backend('ipyparallel'):
    pipeline_wine = Pipeline(
        [
            ('km', KMeans()),
            ('nn',
             MLPClassifier(solver='adam', alpha=1e-5, max_iter=80000, hidden_layer_sizes=(50, 50, 50), random_state=1))
        ]
    )
    param_grid = {'km__n_clusters': range(1, 100, 10)}
    clf = GridSearchCV(pipeline_wine, param_grid, cv=2, verbose=1, n_jobs=5, pre_dispatch=5)
    clf.fit(X_wine, y_wine)  # , n_jobs=16, pre_dispatch=16)
    joblib.dump(clf, 'km_nn_wine.pkl')

if os.path.isfile('em_nn_wine.pkl'):
    clf = joblib.load('em_nn_wine.pkl')
else:
    # #with parallel_backend('ipyparallel'):
    pipeline_wine = Pipeline(
        [
            ('em', GaussianMixtureTransform()),
            ('nn',
             MLPClassifier(solver='adam', alpha=1e-5, max_iter=80000, hidden_layer_sizes=(50, 50, 50), random_state=1))
        ]
    )
    param_grid = {'em__n_components': range(1, 100, 10)}
    clf = GridSearchCV(pipeline_wine, param_grid, cv=2, verbose=1, n_jobs=5, pre_dispatch=5)
    clf.fit(X_wine, y_wine)  # , n_jobs=4, pre_dispatch=1)
    joblib.dump(clf, 'em_nn_wine.pkl')
