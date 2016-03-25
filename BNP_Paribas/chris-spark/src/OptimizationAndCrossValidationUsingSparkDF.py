import os
print 'PYTHONPATH:' + os.environ['PYTHONPATH']

# using get will return `None` if a key is not present rather than raise a `KeyError`
# print os.environ.get('KEY_THAT_MIGHT_EXIST')

import pandas as pd
import numpy as np
import random
import time

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegressionCV
from sklearn.base import BaseEstimator
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV

from pyspark import SparkContext, SparkConf, SQLContext

print ("Successfully imported Spark Modules")

from xgboost import XGBClassifier

start = time.time()

conf = SparkConf().setAppName("Bnp-Paribas").setMaster("local[*]")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sparkContext=sc)


# First ensemble technique (EN_optA)
# Given a set of predictions  X1,X2,...,XnX1,X2,...,Xn , it computes the optimal set of weights w1,w2,...,wnw1,w2,...,wn ;
# and yTyT is the true solution.

def objf_ens_optA(w, Xs, y, n_class):
    #     """
    #     Function to be minimized in the EN_optA ensembler.
    #
    #     Parameters:
    #     ----------
    #     w: array-like, shape=(n_preds)
    #        Candidate solution to the optimization problem (vector of weights).
    #     Xs: list of predictions to combine
    #        Each prediction is the solution of an individual classifier and has a
    #        shape=(n_samples, n_classes).
    #     y: array-like sahpe=(n_samples,)
    #        Class labels
    #     n_class: int
    #        Number of classes in the problem (12 in Airbnb competition)
    #
    #     Return:
    #     ------
    #     score: Scordef objf_ens_optA(w, Xs, y, n_class):
    #     """
    #     Function to be minimized in the EN_optA ensembler.
    #
    #     Parameters:
    #     ----------
    #     w: array-like, shape=(n_preds)
    #     Candidate solution to the optimization problem (vector of weights).
    #     Xs: list of predictions to combine
    #     Each prediction is the solution of an individual classifier and has a
    #     shape=(n_samples, n_classes).
    # y: array-like sahpe=(n_samples,)
    # Class labels
    # n_class: int
    # Number of classes in the problem (12 in Airbnb competition)
    # e of the candidate solution.
    # """
    w = np.abs(w)
    sol = np.zeros(Xs[0].shape)
    for i in range(len(w)):
        sol += Xs[i] * w[i]
    # Using log-loss as objective function (different objective functions can be used here).
    score = log_loss(y, sol)
    return score

class EN_optA(BaseEstimator):
    # """
    # Given a set of predictions $X_1, X_2, ..., X_n$,  it computes the optimal set of weights
    # $w_1, w_2, ..., w_n$; such that minimizes $log\_loss(y_T, y_E)$,
    # where $y_E = X_1*w_1 + X_2*w_2 +...+ X_n*w_n$ and $y_T$ is the true solution.
    # """
    def __init__(self, n_class):
        super(EN_optA, self).__init__()
        self.n_class = n_class

    def fit(self, X, y):
        #     """
        # Learn the optimal weights by solving an optimization problem.
        #
        # Parameters:
        # ----------
        # Xs: list of predictions to be ensembled
        # Each prediction is the solution of an individual classifier and has
        # shape=(n_samples, n_classes).
        # y: array-like
        # Class labels
        # """
        #print X.shape[1], self.n_class

        Xs = np.hsplit(X, X.shape[1]/self.n_class)
        #Initial solution has equal weight for all individual predictions.
        x0 = np.ones(len(Xs)) / float(len(Xs))
        #Weights must be bounded in [0, 1]
        bounds = [(0,1)]*len(x0)
        #All weights must sum to 1
        cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
        #Calling the solver
        res = minimize(objf_ens_optA, x0, args=(Xs, y, self.n_class),
                       method='SLSQP',
                       bounds=bounds,
                       constraints=cons
                       )
        self.w = res.x
        return self

    def predict_proba(self, X):
        # """
        # Use the weights learned in training to predict class probabilities.
        #
        # Parameters:
        # ----------
        # Xs: list of predictions to be blended.
        # Each prediction is the solution of an individual classifier and has
        # shape=(n_samples, n_classes).
        #
        # Return:
        # ------
        # y_pred: array_like, shape=(n_samples, n_class)
        # The blended prediction.
        # """
        Xs = np.hsplit(X, X.shape[1]/self.n_class)
        y_pred = np.zeros(Xs[0].shape)
        for i in range(len(self.w)):
            y_pred += Xs[i] * self.w[i]
        return y_pred

# Second ensemble technique (EN_optB)
# Given a set of predictions X1,X2,...,XnX1,X2,...,Xn , where each Xi has m=12 clases, i.e. Xi=Xi1,Xi2,...,XimXi=Xi1,Xi2,...,Xim .
# The algorithm finds the optimal set of weights w11,w12,...,wnmw11,w12,...,wnm ; such that minimizes
# yTyT is the true solution.
# In [3]:

def objf_ens_optB(w, Xs, y, n_class):
    # """
    # Function to be minimized in the EN_optB ensembler.
    #
    # Parameters:
    # ----------
    # w: array-like, shape=(n_preds)
    # Candidate solution to the optimization problem (vector of weights).
    # Xs: list of predictions to combine
    # Each prediction is the solution of an individual classifier and has a
    # shape=(n_samples, n_classes).
    # y: array-like sahpe=(n_samples,)
    # Class labels
    # n_class: int
    # Number of classes in the problem, i.e. = 12
    #
    # Return:
    # ------
    # score: Score of the candidate solution.
    # """
    # Constraining the weights for each class to sum up to 1.
    # This constraint can be defined in the scipy.minimize function, but doing
    # it here gives more flexibility to the scipy.minimize function
    # (e.g. more solvers are allowed).
    w_range = np.arange(len(w))%n_class
    for i in range(n_class):
        w[w_range==i] = w[w_range==i] / np.sum(w[w_range==i])

    sol = np.zeros(Xs[0].shape)
    for i in range(len(w)):
        sol[:, i % n_class] += Xs[int(i / n_class)][:, i % n_class] * w[i]

    # Using log-loss as objective function (different objective functions can be used here).
    score = log_loss(y, sol)
    return score

class EN_optB(BaseEstimator):
    """
    Given a set of predictions $X_1, X_2, ..., X_n$, where each $X_i$ has
    $m=12$ clases, i.e. $X_i = X_{i1}, X_{i2},...,X_{im}$. The algorithm finds the optimal
    set of weights $w_{11}, w_{12}, ..., w_{nm}$; such that minimizes
    $log\_loss(y_T, y_E)$, where $y_E = X_{11}*w_{11} +... + X_{21}*w_{21} + ...
    + X_{nm}*w_{nm}$ and and $y_T$ is the true solution.
    """
    def __init__(self, n_class):
        super(EN_optB, self).__init__()
        self.n_class = n_class

    def fit(self, X, y):
        # """
        # Learn the optimal weights by solving an optimization problem.
        #
        # Parameters:
        # ----------
        # Xs: list of predictions to be ensembled
        # Each prediction is the solution of an individual classifier and has
        # shape=(n_samples, n_classes).
        # y: array-like
        # Class labels
        # """
        # print X.shape[1], self.n_class

        Xs = np.hsplit(X, X.shape[1]/self.n_class)
        #Initial solution has equal weight for all individual predictions.
        x0 = np.ones(self.n_class * len(Xs)) / float(len(Xs))
        #Weights must be bounded in [0, 1]
        bounds = [(0,1)]*len(x0)
        #Calling the solver (constraints are directly defined in the objective
        #function)
        res = minimize(objf_ens_optB, x0, args=(Xs, y, self.n_class),
                       method='L-BFGS-B',
                       bounds=bounds,
                       )
        self.w = res.x
        return self

    def predict_proba(self, X):
        # """
        # Use the weights learned in training to predict class probabilities.
        #
        # Parameters:
        # ----------
        # Xs: list of predictions to be ensembled
        # Each prediction is the solution of an individual classifier and has
        # shape=(n_samples, n_classes).
        #
        # Return:
        # ------
        # y_pred: array_like, shape=(n_samples, n_class)
        # The ensembled prediction.
        # """

        Xs = np.hsplit(X, X.shape[1]/self.n_class)
        y_pred = np.zeros(Xs[0].shape)
        for i in range(len(self.w)):
            y_pred[:, i % self.n_class] += \
                   Xs[int(i / self.n_class)][:, i % self.n_class] * self.w[i]
        return y_pred

# In [4]:
#
print('Load data...')
DATA_DIR = "../../../../bnp-data"

train = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load(DATA_DIR + '/train_1k.csv')
# train = pd.read_csv(DATA_DIR + "/train.csv")
# plaintext_rdd = sc.textFile(DATA_DIR + "/train_1k.csv")
# train = pycsv.csvToDataFrame(sqlContext, plaintext_rdd)
train.printSchema()

# for x in train.select("*").limit(3):
#     print x

# plaintext_rdd = sc.textFile(DATA_DIR + "/test_1k.csv")
# test = pycsv.csvToDataFrame(sqlContext, plaintext_rdd)
test = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load(DATA_DIR + '/test_1k.csv')

# for x in test.select("*").limit(3):
#     print x

target = train.select("target")

print target

train = train.drop('ID').drop('target')
id_test = test['ID'].values
test = test.drop('ID')

print('Clearing...')

zippedSeries = train.rdd.zip(test.rdd)
zippedSeries.foreach(lambda (train_name, train_series), (test_name, test_series):
    if train_series.dtype == 'O':
    # for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
    test[test_name] = tmp_indexer.get_indexer(test[test_name])
    # but now we have -1 values (NaN)
    else:

    # for int or float: fill NaN
    tmp_len = len(train[train_series.isnull()])
    if tmp_len>0:
    # print "mean", train_series.mean()
        train.loc[train_series.isnull(), train_name] = -9999  # train_series.mean()
    # and Test
    tmp_len = len(test[test_series.isnull()])
    if tmp_len>0:
        test.loc[test_series.isnull(), test_name] = -9999  # train_series.mean()  #TODO
)
# for (train_name, train_series), (test_name, test_series) in zippedSeries.):

# In [5]:

n_classes = 2
for_real = True

# this is what i'll change when i run the whole data set...
# essentially my train and test sets are already split

# Spliting data into train and test sets.
X, X_test, y, y_test = train_test_split(train, target, test_size=0.2)

# Spliting train data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)

print('Data shape:')
print('X_train: %s, X_valid: %s, X_test: %s \n' %(X_train.shape, X_valid.shape,
                                          X_test.shape))

if for_real:
# take the train, target and test data, and come up with a validation set from train
    X_real = train
    X_test_real = test
    y_real = target

    X_train_real, X_valid_real, y_train_real, y_valid_real = train_test_split(X_real, y_real, test_size=0.25)


# Data shape:
# X_train: (68592, 131), X_valid: (22864, 131), X_test: (22865, 131)

# First layer (individual classifiers)
# All classifiers are applied twice: Training on (X_train, y_train) and predicting on (X_valid) Training on (X, y) and predicting on (X_test) You can add / remove classifiers or change parameter values to see the effect on final results.
# In [*]:

# %%time        Replace with timer
# Defining the classifiers
# think about jittering the random state and then averaging the predictions together ...
# only good for extra trees, RF?, native XGB, NN
clfs = {#'LR'  : LogisticRegression(),
    # 'SVM' : SVC(probability=True, random_state=random_state),
    # 'RF'  : RandomForestClassifier(n_estimators=100, n_jobs=-1),
    # 'GBM' : GradientBoostingClassifier(n_estimators=50),
    'ETC' : ExtraTreesClassifier(n_estimators=108, max_features=130, max_depth=12, n_jobs=-1),
    # 'KNN' : KNeighborsClassifier(n_neighbors=30)}
    'XGBc': XGBClassifier(objective='binary:logistic',
                          colsample_bytree=0.77638333498678636,
                          learning_rate=0.030567867858705199,
                          max_delta_step=4.6626180513766657,
                          min_child_weight=57.354121041109124,
                          n_estimators=478,
                          subsample=0.8069399976204783,
                          max_depth=6,
                          gamma=0.2966938071810209)#,
    # 'NN'  : Pipeline([('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
    #                  ('neural network', Classifier(layers=[Layer("Rectifier", units=10),
    #                                                        Layer("Tanh", units=10),
    #                                                        Layer("Softmax")],
    #                                                n_iter=5))])
}

#predictions on the validation and test sets
p_valid = []
p_test = []

p_valid_real = []
p_test_real = []

print('Performance of individual classifiers (1st layer) on X_test')
print('------------------------------------------------------------')

for nm, clf in clfs.items():
    if nm == 'NN':
        #First run. Training on (X_train, y_train) and predicting on X_valid.
        clf.fit(X_train.as_matrix(), y_train)
        yv = clf.predict_proba(X_valid.as_matrix())
        p_valid.append(yv)

        #Second run. Training on (X, y) and predicting on X_test.
        clf.fit(X.as_matrix(), y)
        yt = clf.predict_proba(X_test.as_matrix())
        p_test.append(yt)

        if for_real:
            #First run. Training on (X_train, y_train) and predicting on X_valid.
            clf.fit(X_train_real.as_matrix(), y_train_real)
            yv_real = clf.predict_proba(X_valid_real.as_matrix())
            p_valid_real.append(yv_real)

            #Second run. Training on (X, y) and predicting on X_test.
            clf.fit(X_real.as_matrix(), y_real)
            yt_real = clf.predict_proba(X_test_real.as_matrix())
            p_test_real.append(yt_real)

    else:
        #First run. Training on (X_train, y_train) and predicting on X_valid.
        clf.fit(X_train, y_train)
        yv = clf.predict_proba(X_valid)
        p_valid.append(yv)

        #Second run. Training on (X, y) and predicting on X_test.
        clf.fit(X, y)
        yt = clf.predict_proba(X_test)
        p_test.append(yt)

        if for_real:
            #First run. Training on (X_train, y_train) and predicting on X_valid.
            clf.fit(X_train_real, y_train_real)
            yv_real = clf.predict_proba(X_valid_real)
            p_valid_real.append(yv_real)

            #Second run. Training on (X, y) and predicting on X_test.
            clf.fit(X_real, y_real)
            yt_real = clf.predict_proba(X_test_real)
            p_test_real.append(yt_real)

    #Printing out the performance of the classifier
    print('{:10s} {:2s} {:1.7f}'.format('%s: ' %(nm), 'logloss  =>', log_loss(y_test, yt)))
print('')

#when running the full data
#take out the logloss function... or alternatively, run both the split data and full data model so that
#i can compare my training logloss vs kaggle logloss
# Performance of individual classifiers (1st layer) on X_test
# ------------------------------------------------------------
# ETC:       logloss  => 0.4669750
# XGBc:      logloss  => 0.4657799
#
# CPU times: user 16min 2s, sys: 1.75 s, total: 16min 4s
# Wall time: 11min 8s
# In [*]:

# %%time        Replace with timer
print('MEAN MODELS: Performance of individual classifiers (1st layer) on X_test')
print('------------------------------------------------------------------------')
p_valid_mean = []
p_test_mean = []

p_valid_mean_real = []
p_test_mean_real = []

for nm, clf in clfs.items():
    p_valid_clf = []
    p_test_clf = []
    p_valid_clf_real = []
    p_test_clf_real = []

    holder = []

    for i in range(2):  # 250

        dummy = random.randint(1,10000)
        x = True
        while x == True:
            if dummy in holder:
                dummy = random.randint(1,10000)
            else:
                x = False
        holder.append(dummy)

        random.seed(dummy)

        if nm == 'NN':
            #First run. Training on (X_train, y_train) and predicting on X_valid.
            clf.fit(X_train.as_matrix(), y_train)
            yv = clf.predict_proba(X_valid.as_matrix())
            p_valid_clf.append(yv)

            #Second run. Training on (X, y) and predicting on X_test.
            clf.fit(X.as_matrix(), y)
            yt = clf.predict_proba(X_test.as_matrix())
            p_test_clf.append(yt)

            if for_real:
                #First run. Training on (X_train, y_train) and predicting on X_valid.
                clf.fit(X_train_real.as_matrix(), y_train_real)
                yv_real = clf.predict_proba(X_valid_real.as_matrix())
                p_valid_clf_real.append(yv_real)

                #Second run. Training on (X, y) and predicting on X_test.
                clf.fit(X_real.as_matrix(), y_real)
                yt_real = clf.predict_proba(X_test_real.as_matrix())
                p_test_clf_real.append(yt_real)

        else:
            #First run. Training on (X_train, y_train) and predicting on X_valid.
            clf.fit(X_train, y_train)
            yv = clf.predict_proba(X_valid)
            p_valid_clf.append(yv)

            #Second run. Training on (X, y) and predicting on X_test.
            clf.fit(X, y)
            yt = clf.predict_proba(X_test)
            p_test_clf.append(yt)

            if for_real:
                #First run. Training on (X_train, y_train) and predicting on X_valid.
                clf.fit(X_train_real, y_train_real)
                yv_real = clf.predict_proba(X_valid_real)
                p_valid_clf_real.append(yv_real)

                #Second run. Training on (X, y) and predicting on X_test.
                clf.fit(X_real, y_real)
                yt_real = clf.predict_proba(X_test_real)
                p_test_clf_real.append(yt_real)


    #Printing out the performance of the classifier
    mean_pred_cv = np.mean(p_valid_clf, axis=0)
    mean_pred_test = np.mean(p_test_clf, axis=0)
    p_valid_mean.append(mean_pred_cv)
    p_test_mean.append(mean_pred_test)

    mean_real_pred_cv = np.mean(p_valid_clf_real, axis=0)
    mean_real_pred_test = np.mean(p_test_clf_real, axis=0)
    p_valid_mean_real.append(mean_real_pred_cv)
    p_test_mean_real.append(mean_real_pred_test)

    print('{:10s} {:2s} {:1.7f}'.format('%s - mean: ' %(nm), 'logloss  =>', log_loss(y_test, mean_pred_test)))

print('')

# also try setting different parameters for the XGB and add a NN to the mix
# either use bayesopt for each classifier and putting those into this model, -or-
# randomize both parameters and by random_state... i could do several loops here
# lots of comp time but each random_state run a series of different parameters and
# take the average result for that particular random_state, then run the same
# parameters on the next random_state (don't want random X random as that is hard to replicate)...
# too many combos... let's do bayes_opt

# MEAN MODELS: Performance of individual classifiers (1st layer) on X_test
# ------------------------------------------------------------------------
# In [*]:

# len(XV_mean_real)
len(y_valid)

# In [*]:

print('Performance of optimization based ensemblers (2nd layer) on X_test')
print('------------------------------------------------------------')

#Creating the data for the 2nd layer.
XV = np.hstack(p_valid)
XT = np.hstack(p_test)

n_classes = 2

#EN_optA
enA = EN_optA(n_classes)
enA.fit(XV, y_valid)
w_enA = enA.w
y_enA = enA.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('EN_optA:', 'logloss  =>', log_loss(y_test, y_enA)))

#Calibrated version of EN_optA
cc_optA = CalibratedClassifierCV(enA, method='isotonic')
cc_optA.fit(XV, y_valid)
y_ccA = cc_optA.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optA:', 'logloss  =>', log_loss(y_test, y_ccA)))

#EN_optB
enB = EN_optB(n_classes)
enB.fit(XV, y_valid)
w_enB = enB.w
y_enB = enB.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('EN_optB:', 'logloss  =>', log_loss(y_test, y_enB)))

#Calibrated version of EN_optB
cc_optB = CalibratedClassifierCV(enB, method='isotonic')
cc_optB.fit(XV, y_valid)
y_ccB = cc_optB.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optB:', 'logloss  =>', log_loss(y_test, y_ccB)))
print('')


if for_real:
    print('REAL: Performance of optimization based ensemblers (2nd layer) on X_test')
    print('------------------------------------------------------------')

    #Creating the data for the 2nd layer.
    XV_real = np.hstack(p_valid_real)
    XT_real = np.hstack(p_test_real)

    n_classes = 2

    #EN_optA
    enA_real = EN_optA(n_classes)
    enA_real.fit(XV_real, y_valid_real)
    w_enA_real = enA_real.w
    y_enA_real = enA_real.predict_proba(XT_real)
    #print('{:20s} {:2s} {:1.7f}'.format('EN_optA:', 'logloss  =>', log_loss(y_test, y_enA)))

    #Calibrated version of EN_optA
    cc_optA_real = CalibratedClassifierCV(enA_real, method='isotonic')
    cc_optA_real.fit(XV_real, y_valid_real)
    y_ccA_real = cc_optA_real.predict_proba(XT_real)
    #print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optA:', 'logloss  =>', log_loss(y_test, y_ccA)))

    #EN_optB
    enB_real = EN_optB(n_classes)
    enB_real.fit(XV_real, y_valid_real)
    w_enB_real = enB_real.w
    y_enB_real = enB_real.predict_proba(XT_real)
    #print('{:20s} {:2s} {:1.7f}'.format('EN_optB:', 'logloss  =>', log_loss(y_test, y_enB)))

    #Calibrated version of EN_optB
    cc_optB_real = CalibratedClassifierCV(enB_real, method='isotonic')
    cc_optB_real.fit(XV_real, y_valid_real)
    y_ccB_real = cc_optB_real.predict_proba(XT_real)
    #print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optB:', 'logloss  =>', log_loss(y_test, y_ccB)))
    #print('')

# In [*]:

print('MEAN:  Performance of optimization based ensemblers (2nd layer) on X_test')
print('------------------------------------------------------------')

#Creating the data for the 2nd layer.
XV_mean = np.hstack(p_valid_mean)
XT_mean = np.hstack(p_test_mean)

#EN_optA
enA_mean = EN_optA(n_classes)
enA_mean.fit(XV_mean, y_valid)
w_enA_mean = enA_mean.w
y_enA_mean = enA_mean.predict_proba(XT_mean)
print('{:20s} {:2s} {:1.7f}'.format('EN_optA:', 'logloss  =>', log_loss(y_test, y_enA_mean)))

#Calibrated version of EN_optA
cc_optA_mean = CalibratedClassifierCV(enA_mean, method='isotonic')
cc_optA_mean.fit(XV_mean, y_valid)
y_ccA_mean = cc_optA_mean.predict_proba(XT_mean)
print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optA:', 'logloss  =>', log_loss(y_test, y_ccA_mean)))

#EN_optB
enB_mean = EN_optB(n_classes)
enB_mean.fit(XV_mean, y_valid)
w_enB_mean = enB_mean.w
y_enB_mean = enB_mean.predict_proba(XT_mean)
print('{:20s} {:2s} {:1.7f}'.format('EN_optB:', 'logloss  =>', log_loss(y_test, y_enB_mean)))

#Calibrated version of EN_optB
cc_optB_mean = CalibratedClassifierCV(enB_mean, method='isotonic')
cc_optB_mean.fit(XV_mean, y_valid)
y_ccB_mean = cc_optB_mean.predict_proba(XT_mean)
print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optB:', 'logloss  =>', log_loss(y_test, y_ccB_mean)))
print('')

if for_real:
    print('MEAN:  Performance of optimization based ensemblers (2nd layer) on X_test')
    print('------------------------------------------------------------')

    #Creating the data for the 2nd layer.
    XV_mean_real = np.hstack(p_valid_mean_real)
    XT_mean_real = np.hstack(p_test_mean_real)

    #EN_optA
    enA_mean_real = EN_optA(n_classes)
    enA_mean_real.fit(XV_mean_real, y_valid_real)
    w_enA_mean_real = enA_mean_real.w
    y_enA_mean_real = enA_mean_real.predict_proba(XT_mean_real)
    #print('{:20s} {:2s} {:1.7f}'.format('EN_optA:', 'logloss  =>', log_loss(y_test, y_enA_mean)))

    #Calibrated version of EN_optA
    cc_optA_mean_real = CalibratedClassifierCV(enA_mean_real, method='isotonic')
    cc_optA_mean_real.fit(XV_mean_real, y_valid_real)
    y_ccA_mean_real = cc_optA_mean_real.predict_proba(XT_mean_real)
    #print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optA:', 'logloss  =>', log_loss(y_test, y_ccA_mean)))

    #EN_optB
    enB_mean_real = EN_optB(n_classes)
    enB_mean_real.fit(XV_mean_real, y_valid_real)
    w_enB_mean_real = enB_mean_real.w
    y_enB_mean_real = enB_mean_real.predict_proba(XT_mean_real)
    #print('{:20s} {:2s} {:1.7f}'.format('EN_optB:', 'logloss  =>', log_loss(y_test, y_enB_mean)))

    #Calibrated version of EN_optB
    cc_optB_mean_real = CalibratedClassifierCV(enB_mean_real, method='isotonic')
    cc_optB_mean_real.fit(XV_mean_real, y_valid_real)
    y_ccB_mean_real = cc_optB_mean_real.predict_proba(XT_mean_real)
    #print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optB:', 'logloss  =>', log_loss(y_test, y_ccB_mean)))
    print('')

# Weighted averages
# In [*]:

#come up with better weights here... reflect that in calibration performance
y_3l = (y_enA * 2./9.) + (y_ccA * 4./9.) + (y_enB * 2./9.) + (y_ccB * 1./9.)
print('{:20s} {:2s} {:1.7f}'.format('3rd_layer:', 'logloss  =>', log_loss(y_test, y_3l)))

# In [*]:

#come up with better weights here... reflect that in calibration performance
y_3l_mean = (y_enA_mean * 2./9.) + (y_ccA_mean * 4./9.) + (y_enB_mean * 2./9.) + (y_ccB_mean * 1./9.)
print('{:20s} {:2s} {:1.7f}'.format('3rd_layer:', 'logloss  =>', log_loss(y_test, y_3l_mean)))

# In [ ]:

#top 10% baby!  currently: 155/1942.
# In [ ]:

#optimize weighting of the 3rd level - keep the same weighting for real data
best_score = 10.0

for i in range(10000):
    first = random.randint(0,20)
    second = random.randint(0,20)
    third = random.randint(0,20)
    fourth = random.randint(0,20)
    total = first + second + third + fourth
    first = first / (total * 1.0)
    second = second / (total * 1.0)
    third = third / (total * 1.0)
    fourth = fourth / (total * 1.0)

    y_3l = (y_enA * first) + (y_ccA * second) + (y_enB * third) + (y_ccB * fourth)
    current_score = log_loss(y_test, y_3l)

    if current_score < best_score:
        print('{:20s} {:2s} {:1.7f}'.format('3rd_layer:', 'logloss  =>', log_loss(y_test, y_3l)))
        #print first, second, third, fourth
        best_score = current_score
        best_first = first
        best_second = second
        best_third = third
        best_fourth = fourth
#
# In [ ]:

#optimize weighting of the 3rd level - keep the same weighting for real data
best_mean_score = 10.0

for i in range(10000):
    first = random.randint(0,20)
    second = random.randint(0,20)
    third = random.randint(0,20)
    fourth = random.randint(0,20)
    total = first + second + third + fourth
    first = first / (total * 1.0)
    second = second / (total * 1.0)
    third = third / (total * 1.0)
    fourth = fourth / (total * 1.0)

    y_3l_mean = (y_enA_mean * first) + (y_ccA_mean * second) + (y_enB_mean * third) + (y_ccB_mean * fourth)
    current_score = log_loss(y_test, y_3l_mean)

    if current_score < best_mean_score:
        print('{:20s} {:2s} {:1.7f}'.format('3rd_layer:', 'logloss  =>', log_loss(y_test, y_3l)))
        print first, second, third, fourth
        best_mean_score = current_score
        best_mean_first = first
        best_mean_second = second
        best_mean_third = third
        best_mean_fourth = fourth

# In [ ]:

#well awesome .. CV score is .4577 and my kaggle score is .45373

# In [ ]:

if for_real:
    #preds = (y_enA_real * best_first) + \
    #        (y_ccA_real * best_second) + \
    #        (y_enB_real * best_third) + \
    #        (y_ccB_real * best_fourth)

    preds_mean = (y_enA_mean_real * best_mean_first) + \
            (y_ccA_mean_real * best_mean_second) + \
            (y_enB_mean_real * best_mean_third) + \
            (y_ccB_mean_real * best_mean_fourth)

    pd.DataFrame({"ID": id_test, "PredictedProb": preds_mean[:,1]}).to_csv('3-level_calibrated_model_250iterationsjitteredrandomstate.csv',index=False)

# In [ ]:

#best optimized score i can get with training data is .44956 using untrained XGBClassifier and ExtraTrees,
#jittered 100 random_states with mean predictions with weights [0.0, .9047619047, 0.095238095, 0.0]

#let's try with additional models (logistic regression, random forest, NN), try with maybe randomized params,
#try maybe with bayes_optimized params
# Plotting the weights of each ensemble
# In the case of EN_optA, there is a weight for each prediction and in the case of EN_optB there is a weight for each class for each prediction.
# In [ ]:

end = time.time()
print("TotalExecution Time:")
print(end - start)

from tabulate import tabulate
print('         Weights of EN_optA:')
print('|---------------------------------------|')
wA = np.round(w_enA, decimals=2).reshape(1,-1)
print(tabulate(wA, headers=clfs.keys(), tablefmt="orgtbl"))
print('')
print('     Weights of EN_optB:')
print('|---------------------------|')
wB = np.round(w_enB.reshape((-1,n_classes)), decimals=2)
wB = np.hstack((np.array(list(clfs.keys()), dtype=str).reshape(-1,1), wB))
print(tabulate(wB, headers=['y%s'%(i) for i in range(n_classes)], tablefmt="orgtbl"))

# Comparing our ensemble results with sklearn LogisticRegression based stacking of classifiers.
# Both techniques EN_optA and EN_optB optimizes an objective function. In this experiment I am using the multi-class logloss as objective function. Therefore, the two proposed methods basically become implementations of LogisticRegression. The following code allows to compare the results of sklearn implementation of LogisticRegression with the proposed ensembles.
# In [ ]:


#By default the best C parameter is obtained with a cross-validation approach, doing grid search with
#10 values defined in a logarithmic scale between 1e-4 and 1e4.
#Change parameters to see how they affect the final results.
lr = LogisticRegressionCV(Cs=10, dual=False, fit_intercept=True,
                  intercept_scaling=1.0, max_iter=100,
                  multi_class='ovr', n_jobs=1, penalty='l2',
                  solver='lbfgs', tol=0.0001)

lr.fit(XV, y_valid)
y_lr = lr.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('Log_Reg:', 'logloss  =>', log_loss(y_test, y_lr)))

# In [ ]:


# In [ ]:

print len(p_valid), len(p_valid[0])
print len(np.hstack(p_valid))
# In [ ]:

# for i in range(500):
#
#     dummy = random.randint(1,10000)
#     x = True
#     while x == True:
#         print dummy, str(len(holder)+1), holder
#         #print holder
#         if dummy in holder:
#             dummy = random.randint(1,10000)
#         else:
#             x = False
#         holder.append(dummy)
#
#     random.seed(dummy)

# In [ ]:

#pd.DataFrame({"ID": id_test, "PredictedProb": np.mean(y_pred, axis=0)}).to_csv('extra_trees_and_log_and_gradientboost_with adas_jitteredrandomstate_500iterations.csv',index=False)
