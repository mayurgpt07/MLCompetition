import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
import xgboost as xgb

df = pd.read_csv('C:/Users/Nathan/Desktop/MS Data Science Capstone/norm_dat.csv')

y = df['poverty_rate']
X = df.drop(['poverty_rate'], axis=1)

X = pd.get_dummies(X)
X_final = X.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# gbm = xgb.XGBRegressor(verbose=True, subsample=0.5, n_jobs=4)
svr = SVR()
gbr = GradientBoostingRegressor()
rfr = RandomForestRegressor(n_jobs=-1)
etr = ExtraTreesRegressor(n_jobs=-1)
ada = AdaBoostRegressor()

models = [svr,gbr,rfr,etr,ada]
types = ['svr','gbr','rfr','etr','ada']

# gbm_parameters = {}
svr_parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [100, 1000]}
gbr_parameters = {'n_estimators' :[300, 500, 1000, 1500, 2000], 'max_depth':[4, 6, 8], 'learning_rate': [0.01, 0.02, 0.05, 0.07]}
rfr_parameters = {'n_estimators' :[300, 500, 1000, 1500, 2000], 'max_depth':[4, 6, 8, 10], 'min_samples_split':[2,10,20,40]}
etr_parameters = {'n_estimators' :[300, 500, 1000, 1500, 2000], 'max_depth':[4, 6, 8, 10]}
ada_parameters = {'loss':['square','exponential','linear'],'n_estimators': [300, 500, 1000, 1500, 2000], 'learning_rate': [0.01, 0.02, 0.05, 0.07]}

params = [svr_parameters, gbr_parameters, rfr_parameters, etr_parameters, ada_parameters]

for i in range(5):
    model = models[i]
    parameters = params[i]
    clf = GridSearchCV(model, parameters, cv=5, verbose=True, scoring='neg_mean_squared_error')
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print('\nModel : ', types[i])
    print(clf.best_params_)
    print('\nRoot Mean Square error ', math.sqrt(mean_squared_error(y_test,preds)))
    print('\nExplained Variance ', explained_variance_score(preds,y_test))