import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ANNClassifier(nn.Module):
    def __init__(self):
        super(ANNClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

class FGSM:
    def __init__(self, model, criterion, epsilon=0.3):
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon

    def apply(self, test_loader):
        self.model.eval()
        correct = 0
        adv_examples = []

        for data, target in test_loader:
            data.requires_grad = True
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            if init_pred.item() != target.item():
                continue

            loss = self.criterion(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            sign_data_grad = data_grad.sign()
            perturbed_data = data + self.epsilon*sign_data_grad
            perturbed_data = torch.clamp(perturbed_data, 0, 1)

            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() == target.item():
                correct += 1
            else:
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

        final_acc = correct / float(len(test_loader))
        evasion_rate = 1 - final_acc

        return {
            "evasion_rate": evasion_rate,
            "adv_examples": adv_examples,
            "accuracy": final_acc
        }

# Setup device for training on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def modify_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("layers"):
            new_key = key.replace("layers", "layers.layers")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

if __name__ == "__main__":
    # Load the trained model
    model = ANNClassifier().to(device)
    model_state_dict = torch.load("model.pth", map_location=device)
    
    # Modify keys in the state dictionary
    modified_state_dict = modify_keys(model_state_dict)
    
    # Load model state dictionary
    model.load_state_dict(modified_state_dict)

    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # DataLoader for the MNIST Test set
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Initialize the FGSM attack
    criterion = torch.nn.CrossEntropyLoss()
    fgsm_attack = FGSM(model, criterion, epsilon=0.1)

    # Apply the attack
    results = fgsm_attack.apply(test_loader)

    # Output results
    print("Evasion Rate:", results["evasion_rate"])
    print("Accuracy:", results["accuracy"])

