import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ANNClassifier  # Ensure this is your model's correct import

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_params = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 10
}


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=train_params['batch_size'], shuffle=False)

# Model initialization
model = ANNClassifier(train_params).to(device)

# Defined the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_params['learning_rate'])

# Training loop
model.train()  # Set the model to training mode
for epoch in range(train_params['epochs']):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Evaluation
model.eval()  # Set the model to evaluation mode
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_accuracy = 100. * correct / len(test_loader.dataset)
print(f'Test set: Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.0f}%)')

# Saved the model
torch.save(model.state_dict(), 'model.pth')

