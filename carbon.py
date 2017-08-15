import pickle
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (r2_score, mean_squared_error,
                             explained_variance_score)


SEED = 101
TEST_PROPORTION = 0.1

SUBSET = True
SUBSET_PROPORTION = 0.1

LENSCALE = 10.
SCALAR = False


TR_FILE = "~/Code/carbon/Data_for_Python_V2_wrong_coord_15082017.csv"
QR_FILE = "~/Code/carbon/Grid_NSW_Crop_2015_Python_V2_1482017.csv"
MOD_FILE = "/home/dsteinberg/Code/carbon/GP.pkl"

OUTPUT_FILE = "~/Code/carbon/output.csv"


def negative_log_proba(y_true, y_pred, s_pred):
    nlp = - norm.logpdf(y_true, loc=y_pred, scale=s_pred)
    mean_nlp = np.mean(nlp)
    return mean_nlp


# Load data
def train():
    df = pd.read_csv(TR_FILE)

    # Training and testing split
    y, X = df.iloc[:, 0].values, df.iloc[:, 1:].values
    categorical_cols = [-1]
    Dcon = X.shape[1] - 1
    Dcat = len(set(X[:, -1]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_PROPORTION, random_state=SEED)
    N, D = X_train.shape

    # NOTE: Subsetting
    if SUBSET:
        rand = np.random.RandomState(SEED)
        inds = rand.choice(N, int(N * SUBSET_PROPORTION), replace=False)
        X_train, y_train = X_train[inds], y_train[inds]

    # Initialise estimator
    oh = OneHotEncoder(categorical_features=categorical_cols, sparse=False)
    ss = StandardScaler()

    # Kernel
    l = LENSCALE if SCALAR else LENSCALE * np.ones(Dcon + Dcat)
    kern = 1. * Matern(length_scale=l, nu=2.5) + WhiteKernel(noise_level=0.01)

    # GP
    gp = GaussianProcessRegressor(kern, random_state=SEED)
    gp.predict = partial(gp.predict, return_std=True)  # patch for pipeline

    # Random Forest
    rf = RandomForestRegressor(n_estimators=10, random_state=SEED)

    # SVM
    # http://scikit-learn.org/stable/modules/svm.html
    clf = SVR(C=1.0, epsilon=0.2, kernel='rbf')

    # Linear model
    br = BayesianRidge()

    models = {
        'GP': gp,
        'RandomForest': rf,
        'SVM': clf,
        'BayesianRidge': br
    }
    for name, mod in models.items():
        print("Fitting {}...".format(name))

        if name == 'RandomForest':
            model = mod
        else:
            model = make_pipeline(oh, ss, mod)

        # Train
        model.fit(X_train, y_train)

        # Predict
        if name == 'GP':
            Ey, Sy = model.predict(X_test)
        else:
            Ey = model.predict(X_test)

        # Validate
        r2 = r2_score(y_test, Ey)
        mse = mean_squared_error(y_test, Ey)
        rmse = np.sqrt(mse)
        evs = explained_variance_score(y_test, Ey)

        if name == 'GP':
            nlp = negative_log_proba(y_test, Ey, Sy)
        else:
            nlp = np.inf

        print("{} Results:".format(name))
        print("R2 = {}\nMSE = {}\nRMSE = {}\nEVS = {}\nNLP = {}"
              .format(r2, mse, rmse, evs, nlp))

        if name == 'GP':
            print("Kernel parameters:")
            print(model.steps[-1][1].kernel_)

        with open(name + ".pkl", 'wb') as f:
            pickle.dump(model, f)


def query():
    # Load the query data
    print("Loading data")
    df = pd.read_csv(QR_FILE)
    lon = df["Longitude"]
    lat = df["Latitude"]

    # Load the model
    print("Loading model {}".format(MOD_FILE))
    with open(MOD_FILE, 'rb') as f:
        model = pickle.load(f)

    # Predict
    print("Predicting {} samples.".format(len(df)))
    Ey, Sy = model.predict(df.values)  # GP
    # Ey = model.predict(df.values)  # Not GP

    # Save results
    print("Saving results: {}".format(OUTPUT_FILE))
    res = pd.DataFrame({
        "pred_mean": Ey,
        "pred_std": Sy,
        "Longitude": lon,
        "Latitude": lat
    })
    res.to_csv(OUTPUT_FILE)


if __name__ == "__main__":
    train()
    query()
