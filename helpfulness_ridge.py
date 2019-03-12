import pickle
from joblib import dump

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import time

start = time.time()
with open('data.pkl','rb') as file:
    data = pickle.load(file)
end = time.time()
print("Loaded data.pkl ({:}s)".format(end - start))

# data[['Helpful', 'Unhelpful']].fillna("", inplace=True)
data['Helpful'] = data['HelpfulnessNumerator']
data['Unhelpful'] = data['HelpfulnessDenominator'] - data['Helpful']
df = data.sample(frac=1, random_state=258).reset_index(drop=True)
train, validate, test = np.split(df.sample(frac=1), [int(0.5*len(df)), int(0.75*len(df))])
full = pd.concat([train, validate])
X_train = train[['Helpful', 'Unhelpful']].values
y_train = train['Score'].values
X_validate = validate[['Helpful', 'Unhelpful']].values
y_validate = validate['Score'].values
X_test = test[['Helpful', 'Unhelpful']].values
y_test = test['Score'].values
X_full = full[['Helpful', 'Unhelpful']].values
y_full = full['Score'].values
# X = data[['Helpful', 'Unhelpful']].values
# y = data['Score'].values
# X_full, X_test, y_full, y_test = train_test_split(X, y, test_size=0.20, random_state=258)
# X_train, X_validate, y_train, y_validate = train_test_split(X_full, y_full, test_size=0.25, random_state=258)

reg_params = (0, 0.01, 0.1, 1, 10, 100)
validate_mses = []

## Fit models
for lam in reg_params:
    print("Lambda", lam)
    start = time.time()
    clf = linear_model.Ridge(lam, fit_intercept=False)
    clf.fit(X_train, y_train)
    y_validate_pred = clf.predict(X_validate)
    y_validate_pred[y_validate_pred < 1] = 1
    y_validate_pred[y_validate_pred > 5] = 5
    validate_mse = mean_squared_error(y_validate_pred, y_validate)
    validate_mses.append(validate_mse)
    end = time.time()
    print("  Validate MSE: {} ({:.2}s)".format(validate_mse, end - start))

best_lam, best_validate_mse = reg_params[np.argmin(validate_mses)], min(validate_mses)

print("Best lambda:", best_lam)
print("Best validate MSE:", best_validate_mse)

## Fit test
start = time.time()
best_clf = linear_model.Ridge(best_lam, fit_intercept=False)
best_clf.fit(X_full, y_full)
y_test_pred = best_clf.predict(X_test)
print(min(y_test_pred), max(y_test_pred))
y_test_pred[y_test_pred < 1] = 1
y_test_pred[y_test_pred > 5] = 5
test_mse = mean_squared_error(y_test_pred, y_test)
end = time.time()
print("Test MSE: {} ({:.2}s)".format(test_mse, end - start))

start = time.time()
y_full_pred = best_clf.predict(X_full)
y_full_pred[y_full_pred < 1] = 1
y_full_pred[y_full_pred > 5] = 5
full_mse = mean_squared_error(y_full_pred, y_full)
end = time.time()
print("Full training MSE: {} ({:.2}s)".format(full_mse, end - start))
# dump(vect12, "models/vect12.joblib")
# dump(y_test_pred, "results/vect12_y_test_pred.joblib")
