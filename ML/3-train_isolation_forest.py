import numpy as np, joblib, shap
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from feature_mapping import CICIDS_FEATURES

X_scaled = np.load('X_scaled.npy')
y_binary = np.load('y_binary.npy')

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42
)

iso_forest = IsolationForest(
    n_estimators=200, contamination=0.2,
    random_state=42, n_jobs=-1
)

iso_forest.fit(X_train)

y_pred_raw = iso_forest.predict(X_test)
y_pred = (y_pred_raw == -1).astype(int)

f1 = f1_score(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=['Normal','Attack']))
print(f'F1 Score: {f1:.4f}')

# SHAP — TreeExplainer for Isolation Forest
print('Running SHAP...')
explainer = shap.TreeExplainer(iso_forest)
X_sample = X_test[:200]
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample,
                  feature_names=CICIDS_FEATURES, show=False)
plt.savefig('shap_isolation_forest.png', bbox_inches='tight')

joblib.dump(iso_forest, 'isolation_forest.pkl')
joblib.dump(explainer, 'iso_shap_explainer.pkl')
print('Done. Models and SHAP plot saved.')