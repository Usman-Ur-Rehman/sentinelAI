import numpy as np
import torch
import shap
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import Autoencoder
from feature_mapping import CICIDS_FEATURES

print("🧬 Initializing SHAP DeepExplainer evaluation...")

# Load structural metadata configurations and model weights
config = joblib.load('autoencoder_config.pkl')
model = Autoencoder(config['input_dim'])
model.load_state_dict(torch.load('autoencoder.pth'))
model.eval()

X_scaled = np.load('ml/X_scaled.npy')

# Use 100 samples of pure normal traffic as baseline background context
background = torch.FloatTensor(X_scaled[:100])
test_sample = torch.FloatTensor(X_scaled[100:300])

# Configure the DeepExplainer instance
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_sample)

# 🐛 Bug 4 Fix: shap_values is returned as a list of arrays (one array per output neuron).
# In an Autoencoder, output_dim == input_dim. If you index only shap_values[0], you explain
# only the first feature's reconstruction. To obtain the overall feature importance across
# all output dimensions, we compute the mean absolute SHAP values across axis=0.
shap_array = np.array(shap_values)  # Shape: (output_dim, n_samples, n_features)
shap_mean_abs = np.mean(np.abs(shap_array), axis=0)  # Shape: (n_samples, n_features)

print("📈 Generating SHAP summary plots...")
shap.summary_plot(
    shap_mean_abs,
    test_sample.numpy(),
    feature_names=CICIDS_FEATURES,
    show=False
)
plt.savefig('ml/shap_autoencoder.png', bbox_inches='tight')
plt.close()

# Save background matrices for explainability pipelines inside runtime servers
joblib.dump(background.numpy(), 'ml/ae_shap_background.npy')
print("✅ SHAP Autoencoder visualization saved successfully. [Bug 4 Corrected]")