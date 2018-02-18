# MLCompetition
Code used to win 2nd place out of 453 students in the Microsoft Data Science Professional Certification Capstone.
 
This competition was focused on predicting poverty rates (a regression problem) for counties in the United States based on 33 categorical and continuous features about those counties.
 
More information about the competition can be found here: https://www.datasciencecapstone.org/competitions/3/county-poverty/
 
Six regression models  (multi-layer perceptron, support vector machine, gradient boosted regressor, random forest regressor, extra trees regressor, and adaboost regresor) were tuned using grid search cross-validation against the training dataset to optimize mean squared error.
 
The predictions for each model were used as features to train a final regression model (XGBoost regresor) using the stacking technique with the library vecstack. The model was evaluated on a held-out sample of the training set. Once proven to be effective with the selected parameters, the model was then re-trained on the entire training set to reduce bias.
 
The trained final model was then applied to the test dataset resulting in a test root mean squared error of 2.57. A report summarizing the exploration of the dataset, modeling procedure, and suggested actions was then created.
