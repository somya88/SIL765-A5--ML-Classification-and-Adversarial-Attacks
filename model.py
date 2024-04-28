import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ANNClassifier(nn.Module):
    def __init__(self, train_params: dict):
        super(ANNClassifier, self).__init__()
        self.train_params = train_params
        self._define_model()
        self.criterion = self._define_criterion()
        self.optimizer = self._define_optimizer()

    def _define_model(self) -> None:
        # Define the layers of the model here
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.activation = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def _define_criterion(self) -> nn.Module:
        # Define the criterion (loss function)
        return nn.NLLLoss()

    def _define_optimizer(self) -> torch.optim.Optimizer:
        # Define the optimizer
        return optim.Adam(self.parameters(), lr=self.train_params['learning_rate'])

    def forward(self, x):
        # Forward pass through the network
        x = x.view(-1, 28*28)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def train_step(self, train_loader):
        # Training loop
        self.train()  # Set the model to training mode
        for epoch in range(self.train_params['epochs']):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                output = self.forward(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                
            print(f"Epoch {epoch+1}/{self.train_params['epochs']} completed.")

        # Here you can return anything you need for reporting and plotting
        return {'loss': loss.item()}

    def infer(self, test_loader):
        # Evaluate the model
        self.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def plot_loss(self, results: dict) -> None:
        # Implement plotting logic here if required for the assignment
        pass

    def save(self, file_path: str):
        # Save the model's state_dict
        torch.save(self.state_dict(), file_path)

# Setup device for training on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example usage
if __name__ == '__main__':
    train_params = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 10
    }
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    # Create dataloaders
    def get_dataloaders(batch_size):
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader

    train_loader, test_loader = get_dataloaders(train_params['batch_size'])

    # Create model
    model = ANNClassifier(train_params).to(device)

    # Train and evaluate
    results = model.train_step(train_loader)
    model.save('model.pth')
    test_accuracy = model.infer(test_loader)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

