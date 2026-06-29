import numpy as np
import torch
import shap
import joblib
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend safe for terminal execution
import matplotlib.pyplot as plt

# Programmatically resolve import paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# Ensure our target output directory exists (handles 'ML' vs 'ml' case sensitivity on Linux)
os.makedirs(SCRIPT_DIR, exist_ok=True)

# Helper function to generate absolute paths relative to the script location
def get_path(filename):
    return os.path.join(SCRIPT_DIR, filename)

from models import Autoencoder
from feature_mapping import CICIDS_FEATURES

print("🧬 Initializing SHAP DeepExplainer evaluation...")

# Load structural metadata configurations and model weights
config = joblib.load(get_path('autoencoder_config.pkl'))
model = Autoencoder(config['input_dim'])
model.load_state_dict(torch.load(get_path('autoencoder.pth')))
model.eval()

X_scaled = np.load(get_path('X_scaled.npy'))

# Use 100 samples of pure normal traffic as baseline background context
background = torch.FloatTensor(X_scaled[:100])
test_sample = torch.FloatTensor(X_scaled[100:300])

# Configure the DeepExplainer instance
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_sample)

# 🐛 Bug 4 Fix: shap_values is returned as a list of arrays (one array per output neuron).
# We want overall feature importance across ALL output neurons.
# Compute mean absolute SHAP across all output neurons.
shap_array = np.array(shap_values)  # Shape: (output_dim, n_samples, n_features)
shap_mean_abs = np.mean(np.abs(shap_array), axis=0)  # Shape: (n_samples, n_features)

print("📈 Generating SHAP summary plots...")
plt.figure(figsize=(10, 6))

# To avoid the assertion error, we generate a global bar plot of feature importances.
# This represents the mean absolute SHAP value for each feature across all samples.
global_importances = np.mean(shap_mean_abs, axis=0)
indices = np.argsort(global_importances)

# Plot horizontal bar chart
plt.title("SHAP Autoencoder Global Feature Importance")
plt.barh(range(len(indices)), global_importances[indices], color='indigo', align='center')
plt.yticks(range(len(indices)), [CICIDS_FEATURES[i] for i in indices])
plt.xlabel("mean(|SHAP value|) (average impact on model reconstruction)")
plt.tight_layout()

plt.savefig(get_path('shap_autoencoder.png'), bbox_inches='tight')
plt.close()

# Save background matrices for explainability pipelines inside runtime servers
joblib.dump(background.numpy(), get_path('ae_shap_background.npy'))
print("✅ SHAP Autoencoder visualization saved successfully. [Bug 4 Corrected & Multi-output index aligned]")
# ```
# eof

# ### Why This Resolves the Bug:
# * Instead of letting `shap.summary_plot` try to interpret the 2D averaged array (which triggers internal row-matching checks designed for single-output classification), we directly calculate the **global feature importances** (`global_importances = np.mean(shap_mean_abs, axis=0)`).
# * We then construct a clean, native matplotlib bar chart sorted by feature impact. This is mathematically identical to what a SHAP global bar summary plot does, but bypasses the legacy internal check.
# * It saves the file safely as `shap_autoencoder.png` in your `ml`/`ML` directory.

# Go ahead and run this updated script with your active virtual environment:
# ```bash
# python ml/shap_autoencoder.py