import numpy as np
from sklearn.base import clone
from autosklearn.classification import AutoSklearnClassifier
import pandas as pd
from autosklearn.classification import AutoSklearnClassifier


if __name__ == "__main__":
    
    def fit(model, X, labels):
        models = [clone(model).fit(X,labels[i]) for i in range(0,len(labels))]
        return(models)

    def predict(models, X):
        predictions = [m.predict_proba(X)[:,1] for m in models]
        return(predictions)

    # def evaluate(y, predictions):
    #     # predictions = np.where(np.array(predictions) > 0.5, 1, 0)
    #     predictions = np.array(predictions)
    #     y = np.array(y)
    #      return [roc_auc_score(y[:, i], predictions[i, :]) for i in range(y.shape[1])]

    def save_predictions(predictions, filename):
        df = pd.DataFrame(np.array(predictions).T, columns=['attribute 1', 'attribute 2', 'attribute 3', 'attribute 4', 'attribute 5'])
        df.to_csv(filename, index_label= 'Id')


    model = AutoSklearnClassifier(
      time_left_for_this_task = 45 * 60,
      per_run_time_limit = 125,
      n_jobs = -1
    )

    X_train = np.genfromtxt('protein_features_train.csv', delimiter=',')
    y_train = np.genfromtxt('protein_labels_train.csv', delimiter=',')
    X_test = np.genfromtxt('protein_features_test.csv', delimiter=',')

    labels_train = [np.copy(y_train[:,i]) for i in range(0, y_train.shape[1])]

    models = fit(model, X_train, labels_train)
    predictions = predict(models, X_test)
    save_predictions(predictions, 'predictions.csv')
