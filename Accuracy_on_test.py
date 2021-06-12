from share_data80_20 import y_train, y_test
from Preprocessing_data import X_train, X_test
from base_network_model import model
from learning_similarity import _most_similarity
from sklearn.metrics import accuracy_score
X_train_vec = model.predict(X_train)
X_test_vec = model.predict(X_test)
y_preds = []
for vec in X_test_vec:
  vec = vec.reshape(1, -1)
  y_pred = _most_similarity(X_train_vec, vec, y_train)
  y_preds.append(y_pred)

print(accuracy_score(y_preds, y_test))