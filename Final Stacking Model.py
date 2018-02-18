import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from vecstack import stacking

# Load data and convert categorical features to onehot
df = pd.read_csv('C:/Users/Nathan/Desktop/MS Data Science Capstone/norm_dat.csv')

y = df['poverty_rate']
X = df.drop(['poverty_rate'], axis=1)

X = pd.get_dummies(X)
X_final = X.as_matrix()
y_final = y.as_matrix()

# Make train/test split
# As usual in machine learning tasks we have X_train, y_train, and X_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize 1-st level models.
models = [
    ExtraTreesRegressor(random_state=0, n_jobs=-1, n_estimators=1500, max_depth=10),

    RandomForestRegressor(random_state=0, n_jobs=-1, n_estimators=1000, max_depth=8),

    GradientBoostingRegressor(learning_rate=0.05, n_estimators=2000, max_depth=6),

    SVR(C=1000, gamma=0.001, kernel='rbf'),

    MLPRegressor(hidden_layer_sizes=(300,300,300,64), activation='relu', solver='lbfgs', alpha=0.0001,
                                         batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                                         power_t=0.5, max_iter=400, shuffle=True, random_state=None,
                                         tol=0.0001, verbose=True, warm_start=False, momentum=0.8,
                                         nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                                         beta_1=0.9, beta_2=0.999, epsilon=1e-08),

    AdaBoostRegressor(loss='square',learning_rate=0.07, n_estimators=2000)]

# Compute stacking features
S_train, S_test = stacking(models, X_train, y_train, X_test,
                           regression=True, metric=mean_squared_error, n_folds=4,
                           shuffle=True, random_state=0, verbose=2)

# Initialize 2-nd level model
model = XGBRegressor(seed=0, n_jobs=-1)

parameters = {'max_depth': [4,6,8], 'n_estimators': [700, 1000, 2000, 2500, 3000], 'learning_rate' : [0.01, 0.02, 0.03, 0.05]}

gs_model = GridSearchCV(model, parameters, cv=5, verbose=True, scoring='neg_mean_squared_error')

# Fit 2-nd level model
gs_model.fit(S_train, y_train)

# Predict
y_pred = gs_model.predict(S_test)

# Final prediction score
print('\nRoot Mean Square error ', math.sqrt(mean_squared_error(y_test,y_pred)))
print('\nExplained Variance ', explained_variance_score(y_test, y_pred))

# Import the real test data into a numpy array, fixing categorical features
df_test = pd.read_csv('C:/Users/Nathan/Desktop/MS Data Science Capstone/test_dat.csv')
df_test = pd.get_dummies(df_test)
test_final = df_test.as_matrix()

# Run on the real data
S_train_final, S_test_final = stacking(models, X_final, y_final, test_final,
                           regression=True, metric=mean_squared_error, n_folds=4,
                           shuffle=True, random_state=0, verbose=2)

final_model = GridSearchCV(model, parameters, cv=5, verbose=True, scoring='neg_mean_squared_error')

# Fit 2-nd level model
final_model.fit(S_train_final, y_final)

# Predict and save predictions
y_pred = final_model.predict(S_test_final)

np.savetxt("C:/Users/Nathan/Desktop/MS Data Science Capstone/stacking_scores.csv", y_pred, delimiter=",")
print('Done')