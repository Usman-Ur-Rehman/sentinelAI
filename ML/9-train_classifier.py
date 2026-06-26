import numpy as np, pandas as pd, joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

X_scaled = np.load('X_scaled.npy')
y_multi = pd.read_csv('y_multiclass.csv').squeeze()

le = LabelEncoder()
y_encoded = le.fit_transform(y_multi)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2,
    random_state=42, stratify=y_encoded
)

clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='mlogloss',
    n_jobs=-1,
    random_state=42
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le.classes_))
print('Macro F1:', round(f1_score(y_test, y_pred, average='macro'), 4))

joblib.dump(clf, 'attack_classifier.pkl')
joblib.dump(le, 'label_encoder.pkl')
print('Classifier saved.')