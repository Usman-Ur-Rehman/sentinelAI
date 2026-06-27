import pandas as pd, numpy as np, glob, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_mapping import CICIDS_FEATURES

csv_files = glob.glob('ML/cicids/*.csv')
dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
data = pd.concat(dfs, ignore_index=True)
data.columns = data.columns.str.strip()
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

X = data[CICIDS_FEATURES]
y_binary = (data['Label'] != 'BENIGN').astype(int)
y_multiclass = data['Label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, 'ML/scaler.pkl')
np.save('ML/X_scaled.npy', X_scaled)
np.save('ML/y_binary.npy', y_binary.values)
pd.Series(y_multiclass.values).to_csv('ML/y_multiclass.csv', index=False)

print('Data prep complete. Shape:', X_scaled.shape)