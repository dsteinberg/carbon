from functools import partial
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (r2_score, mean_squared_error,
                             explained_variance_score)


SEED = 101
TEST_PROPORTION = 0.1

SUBSET = False
SUBSET_PROPORTION = 0.1

LENSCALE = 1.
SCALAR = False


# FILE = "C:/Users/Senani_2/Documents/Data_61_project_2017/To_Daniel_17072017/" \
#        "to_python/stacked_data_for_modelling_to_python_17072017.csv"
FILE = "~/Code/carbon/stacked_data_for_modelling_to_GIS_with" \
       "_moved_coords_with_discrete_variables_V3_20072017.csv"


def negative_log_proba(y_true, y_pred, s_pred):
    nlp = - norm.logpdf(y_true, loc=y_pred, scale=s_pred)
    mean_nlp = np.mean(nlp)
    return mean_nlp


# Load data
df = pd.read_csv(FILE)

# Training and testing split
y, X = df.iloc[:, 0].values, df.iloc[:, 1:-2].values
Dcont = X.shape[1]

# Onehot data
soil = pd.get_dummies(df.soil)
landuse = pd.get_dummies(df.landuse, drop_first=True)  # Binary
Dcat = soil.shape[1] + landuse.shape[1]

# Stack all X
X = np.hstack((X, soil, landuse))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_PROPORTION, random_state=SEED)
N, D = X_train.shape

# NOTE: Subsetting
if SUBSET:
    rand = np.random.RandomState(SEED)
    inds = rand.choice(N, int(N * SUBSET_PROPORTION), replace=False)
    X_train, y_train = X_train[inds], y_train[inds]

# Initialise estimator
ss = StandardScaler()

# Kernel
l = LENSCALE if SCALAR else LENSCALE * np.ones(D)
kern = 1. * Matern(length_scale=l, nu=2.5) + WhiteKernel(noise_level=0.1)

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
    model = make_pipeline(ss, mod)

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
        print(model.steps[1][1].kernel_)
