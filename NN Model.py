import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import math

df = pd.read_csv('F:/Downloads/norm_dat.csv')

y = df['poverty_rate']
X = df.drop(['poverty_rate'], axis=1)

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = MLPRegressor(hidden_layer_sizes=(1000,1000,1000,64), activation='relu', solver='lbfgs', alpha=0.0001,
                                         batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                                         power_t=0.5, max_iter=400, shuffle=True, random_state=None,
                                         tol=0.0001, verbose=True, warm_start=False, momentum=0.8,
                                         nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                                         beta_1=0.9, beta_2=0.999, epsilon=1e-08)


model.fit(X_train, y_train)
print(math.sqrt(mean_squared_error(y_test, model.predict(X_test))))

exit()

X = X.as_matrix()
y = y.as_matrix()

model.fit(X, y)

test = pd.read_csv('F:/Downloads/NN_test.csv')
test2 = test.iloc[:,1:].as_matrix()

results = pd.Series(model.predict(test2))

results.to_csv('F:/Downloads/NN_attempt_2.5.csv')
test['row_id'].to_csv('F:/Downloads/NN_attempt_2.csv')



