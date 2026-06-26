import numpy as np
import torch
from torch import nn
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, classification_report
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend safe for terminal execution
import matplotlib.pyplot as plt
from models import AutoEncoder

print("🧠 Loading preprocessed training matrices...")
X_scaled=np.load("x_scaled.npy")
Y_binary=np.load("y_binary.npy")

#extracting only the normal packets and converting to 32 bit floating point tensor
X_normal=X_scaled[Y_binary==0]
X_normal_tensor=torch.FloatTensor(X_normal)

#data loader configured to give data in batches
loader=DataLoader(TensorDataset(X_normal_tensor),batch_size=256,shuffle=True)
input_dim=X_scaled.shape[1]

#making model and selecting optimizer and updation criteria
model=AutoEncoder(input_dim)
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
criterion=nn.MSELoss()

#looping he model to run on batches and then for each batch in an epoch calculate
#the loss and optimize the model
for epoch in range(50):
    model.train()
    total_loss=0
    for(x,) in loader:
        pred=model(x)
        loss=criterion(pred,x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

    if (epoch + 1) % 10 == 0:
        print(f" ── 📈 Epoch {epoch+1:02d}/50 | Reconstruction Loss: {total_loss/len(loader):.6f}")

model.eval()

with torch.no_grad():
    X_all = torch.FloatTensor(X_scaled)
    reconstructed_out = model(X_all)
    # Calculate Mean Squared Error (MSE) reconstruction loss per event
    errors = torch.mean((X_all - reconstructed_out)**2, dim=1).numpy()

# Plot the physical distribution curve to inspect normal vs attack separation boundaries
print("📊 Generating error distribution plot...")
plt.figure(figsize=(10, 5))
plt.hist(errors[Y_binary == 0], bins=50, alpha=0.7, label='Normal Traffic', color='green')
plt.hist(errors[Y_binary == 1], bins=50, alpha=0.7, label='Attack Traffic', color='red')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Count')
plt.title('Autoencoder Reconstruction Error — Normal vs Attack')
plt.legend()
plt.savefig('ml/autoencoder_error_distribution.png', bbox_inches='tight')
plt.close()

# Calculate optimal decision boundary threshold using Youden's J-statistic
fpr, tpr, thresholds = roc_curve(Y_binary, errors)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = float(thresholds[optimal_idx])
print(f"🎯 Optimal reconstruction error threshold: {optimal_threshold:.6f}")

# Apply threshold classification and display performance report
y_pred_ae = (errors > optimal_threshold).astype(int)
print("\n📋 Autoencoder Performance Metrics against Ground Truth:")
print(classification_report(Y_binary, y_pred_ae, target_names=['Normal', 'Attack']))

# Serialize final state configurations for runtime hot-swaps
print("📦 Exporting PyTorch state dictionary weights and threshold profiles...")
torch.save(model.state_dict(), 'ml/autoencoder.pth')
joblib.dump({'threshold': optimal_threshold, 'input_dim': input_dim}, 'ml/autoencoder_config.pkl')
np.save('ml/reconstruction_errors.npy', errors)
print("🏁 Done. Autoencoder fully trained and serialized.")