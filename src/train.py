import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.model import GravitationalWaveAutoencoder

def train_autoencoder(epochs=5, batch_size=32):
    model = GravitationalWaveAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Generate synthetic background noise dataset for training
    print("Generating training background noise data...")
    dummy_dataset = np.random.normal(0, 0.5, (100, 1, 4096)) # 100 samples
    dataset_tensor = torch.tensor(dummy_dataset, dtype=torch.float32)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(dataset_tensor)
        loss = criterion(outputs, dataset_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
    # Save trained weights
    torch.save(model.state_dict(), "gw_autoencoder.pth")
    print("Model training complete and weights saved to gw_autoencoder.pth!")

if __name__ == "__main__":
    train_autoencoder()