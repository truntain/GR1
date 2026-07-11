import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. Thiet lap cau hinh va thu muc
os.makedirs('output', exist_ok=True)
DATA_DIR = './data'
epochs = 5
batch_size = 64
learning_rate = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dang su dung thiet bi: {device}")

# 2. Chuan bi du lieu (MNIST Dataset)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Chuan hoa du lieu MNIST trung binh va do lech chuan
])

print("Dang tai va chuan bi bo du lieu MNIST...")
train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
print(f"Tai thanh cong! Co {len(train_dataset)} mau train va {len(test_dataset)} mau test.")

# 3. Dinh nghia mang no-ron MLP (Multi-Layer Perceptron) nhu da hoc
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        # Tang an thu 1: 784 -> 128
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Ham kich hoat phi tuyen ReLU
        self.relu = nn.ReLU()
        # Tang dau ra: 128 -> 10 (chua di qua Softmax vi CrossEntropyLoss trong PyTorch da tich hop)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Flatten anh tu (batch_size, 1, 28, 28) thanh (batch_size, 784)
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = MLP().to(device)
print(model)

# 4. Thiet lap Ham chi phi (Loss) va Bo toi uu (Optimizer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 5. Vong lap huan luyen (Training Loop)
train_losses = []
train_accuracies = []

print("\n--- BAT DAU HUAN LUYEN MODEL ---")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # 1. Lan truyen xuoi (Forward Propagation)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 2. Xoa cac dao ham cu (Zero gradients)
        optimizer.zero_grad()
        
        # 3. Lan truyen nguoc (Backward Propagation)
        loss.backward()
        
        # 4. Cap nhat cac tham so (Gradient Descent step)
        optimizer.step()
        
        # Thong ke loss va accuracy
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = (correct / total) * 100
    epoch_time = time.time() - start_time
    
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    print(f"==> Epoch {epoch+1} hoan thanh trong {epoch_time:.2f} giay | Loss trung binh: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

print("\n--- HUAN LUYEN HOAN TAT ---")

# 6. Danh gia tren tap kiem thu (Test dataset)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Do chinh xac tren tap kiem thu (Test Accuracy): {(correct / total) * 100:.2f}%\n")

# 7. Ve do thi ket qua va luu vao thu muc output
print("Dang ve do thi bieu dien ket qua...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Do thi Loss
ax1.plot(range(1, epochs + 1), train_losses, marker='o', color='red', label='Train Loss')
ax1.set_title('Bieu do Loss qua cac Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)
ax1.legend()

# Do thi Accuracy
ax2.plot(range(1, epochs + 1), train_accuracies, marker='s', color='blue', label='Train Accuracy')
ax2.set_title('Bieu do Accuracy qua cac Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
output_path = 'output/training_result.png'
plt.savefig(output_path)
print(f"Luu do thi ket qua thanh cong tai: {output_path}")
print("Tien trinh hoan tat!")
