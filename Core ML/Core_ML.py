import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyper-parameters
num_epochs = 12
batch_size = 10
learning_rate_ce = 0.005  # Higher learning rate for CE
learning_rate_nce = 0.001  # Lower learning rate for NCE and APL
noise_rates = [0.3, 0.5, 0.7]

apl_params = {
    0.3: {'alpha': 1.4, 'beta': 0.6},
    0.5: {'alpha': 1.0, 'beta': 1.0},
    0.7: {'alpha': 0.6, 'beta': 1.4}
}

# Create directory for results
os.makedirs('results', exist_ok=True)

# Dataset transformations with data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR10 dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class ConvNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(F.relu(self.bn3(self.fc1(x))))
        x = self.fc2(x)
        return x

class NormalizedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(NormalizedCrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets):
        # Calculating standard cross entropy
        ce = self.ce_loss(outputs, targets)

        num_classes = outputs.size(1)
        target_onehot = F.one_hot(targets, num_classes).float()
        target_dist = torch.mean(target_onehot, dim=0)

        target_dist = torch.clamp(target_dist, min=1e-6)
        base_level = -torch.sum(target_dist * torch.log(target_dist))

        # Adding a minimum threshold to base_level to prevent division by very small numbers
        base_level = torch.clamp(base_level, min=0.1)

        # Normalizing the cross entropy by the base level
        nce = ce / base_level

        return torch.mean(nce)


# Active-Passive Loss using NCE + MAE
class APLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(APLoss, self).__init__()
        self.alpha = alpha  # Weight for active loss (NCE)
        self.beta = beta  # Weight for passive loss (MAE)
        self.nce = NormalizedCrossEntropyLoss()

    def forward(self, outputs, targets):
        # Active loss: Normalized Cross-Entropy
        active_loss = self.nce(outputs, targets)

        # Passive loss: Mean Absolute Error
        num_classes = outputs.size(1)
        probs = F.softmax(outputs, dim=1)
        target_onehot = F.one_hot(targets, num_classes).float()
        passive_loss = torch.mean(torch.abs(probs - target_onehot))

        # Combining losses
        total_loss = self.alpha * active_loss + self.beta * passive_loss

        return total_loss


def apply_symmetric_noise(dataset, eta):
    num_classes = 10
    targets = np.array(dataset.targets)
    num_samples = len(targets)
    num_noise = int(num_samples * eta)

    # Indices to add noise to
    noise_indices = random.sample(range(num_samples), num_noise)

    for idx in noise_indices:
        # Generate all possible incorrect labels (excluding the true label)
        incorrect_labels = list(range(num_classes))
        incorrect_labels.remove(targets[idx])
        # Randomly select one incorrect label
        new_label = random.choice(incorrect_labels)
        targets[idx] = new_label

    # Update the dataset targets
    dataset.targets = targets.tolist()
    return dataset


# Function to evaluate model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


# Function to train model
def train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, epochs):
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 1000 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        test_acc = evaluate_model(model, test_loader)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        test_accuracies.append(test_acc)

        print(
            f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, Test Accuracy: {test_acc:.2f}%')

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr}')

    return train_losses, train_accuracies, test_accuracies


# Function to run experiment for a specific noise rate
def run_experiment_for_noise_rate(noise_rate):
    print(f"\n\n==== Running experiment with noise rate: {noise_rate} ====\n")

    # Get APL parameters for this noise rate
    alpha = apl_params[noise_rate]['alpha']
    beta = apl_params[noise_rate]['beta']
    print(f"APL parameters: alpha={alpha}, beta={beta}")

    # Prepare datasets
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform_train)

    # Applying noise
    noisy_dataset = apply_symmetric_noise(train_dataset, noise_rate)
    train_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=batch_size, shuffle=True)

    results = {}

    # Train with standard Cross-Entropy Loss
    print("\nTraining with standard Cross-Entropy Loss:")
    model_ce = ConvNet().to(device)
    criterion_ce = nn.CrossEntropyLoss()
    optimizer_ce = torch.optim.Adam(model_ce.parameters(), lr=learning_rate_ce, weight_decay=1e-4)
    scheduler_ce = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_ce, milestones=[4,8], gamma=0.25
    )
    ce_train_losses, ce_train_accs, ce_test_accs = train_model(
        model_ce, criterion_ce, optimizer_ce, scheduler_ce, train_loader, test_loader, num_epochs)

    results['CE'] = {
        'train_losses': ce_train_losses,
        'train_accuracies': ce_train_accs,
        'test_accuracies': ce_test_accs
    }

    # Train with Normalized Cross-Entropy Loss
    print("\nTraining with Normalized Cross-Entropy Loss:")
    model_nce = ConvNet().to(device)
    criterion_nce = NormalizedCrossEntropyLoss()
    optimizer_nce = torch.optim.Adam(model_nce.parameters(), lr=learning_rate_nce, weight_decay=1e-4)
    scheduler_nce = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_nce, milestones=[4,8], gamma=0.25
    )
    nce_train_losses, nce_train_accs, nce_test_accs = train_model(
        model_nce, criterion_nce, optimizer_nce, scheduler_nce, train_loader, test_loader, num_epochs)

    results['NCE'] = {
        'train_losses': nce_train_losses,
        'train_accuracies': nce_train_accs,
        'test_accuracies': nce_test_accs
    }

    # Train with Active-Passive Loss (NCE + MAE)
    print(f"\nTraining with Active-Passive Loss (NCE + MAE, alpha={alpha}, beta={beta}):")
    model_apl = ConvNet().to(device)
    criterion_apl = APLoss(alpha=alpha, beta=beta)
    optimizer_apl = torch.optim.Adam(model_apl.parameters(), lr=learning_rate_nce, weight_decay=1e-4)
    scheduler_apl = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_apl, milestones=[4,8], gamma=0.25
    )
    apl_train_losses, apl_train_accs, apl_test_accs = train_model(
        model_apl, criterion_apl, optimizer_apl, scheduler_apl, train_loader, test_loader, num_epochs)

    results['APL'] = {
        'train_losses': apl_train_losses,
        'train_accuracies': apl_train_accs,
        'test_accuracies': apl_test_accs,
        'params': {'alpha': alpha, 'beta': beta}
    }

    # Plot results for this noise rate
    plot_results_for_noise_rate(results, noise_rate)

    return results


# Function to plot results for a specific noise rate
def plot_results_for_noise_rate(results, noise_rate):
    epochs = range(1, num_epochs + 1)
    alpha = results['APL']['params']['alpha']
    beta = results['APL']['params']['beta']

    # Save the data to a text file
    with open(f'results/noise_rate_{noise_rate}_results.txt', 'w') as f:
        f.write(
            "Epoch, CE Train Loss, NCE Train Loss, APL Train Loss, CE Train Acc, NCE Train Acc, APL Train Acc, CE Test Acc, NCE Test Acc, APL Test Acc\n")
        for i in range(len(epochs)):
            f.write(
                f"{epochs[i]}, {results['CE']['train_losses'][i]}, {results['NCE']['train_losses'][i]}, {results['APL']['train_losses'][i]}, "
                f"{results['CE']['train_accuracies'][i]}, {results['NCE']['train_accuracies'][i]}, {results['APL']['train_accuracies'][i]}, "
                f"{results['CE']['test_accuracies'][i]}, {results['NCE']['test_accuracies'][i]}, {results['APL']['test_accuracies'][i]}\n")

    try:
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot losses
        ax1.plot(epochs, results['CE']['train_losses'], 'b-', label='CE Loss')
        ax1.plot(epochs, results['NCE']['train_losses'], 'r-', label='NCE Loss')
        ax1.plot(epochs, results['APL']['train_losses'], 'g-', label=f'APL Loss (α={alpha}, β={beta})')
        ax1.set_title(f'Training Loss (Noise Rate = {noise_rate})')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot accuracies
        ax2.plot(epochs, results['CE']['train_accuracies'], 'b--', label='CE Train Acc')
        ax2.plot(epochs, results['NCE']['train_accuracies'], 'r--', label='NCE Train Acc')
        ax2.plot(epochs, results['APL']['train_accuracies'], 'g--', label=f'APL Train Acc (α={alpha}, β={beta})')
        ax2.plot(epochs, results['CE']['test_accuracies'], 'b-', label='CE Test Acc')
        ax2.plot(epochs, results['NCE']['test_accuracies'], 'r-', label='NCE Test Acc')
        ax2.plot(epochs, results['APL']['test_accuracies'], 'g-', label=f'APL Test Acc (α={alpha}, β={beta})')
        ax2.set_title(f'Accuracy (Noise Rate = {noise_rate})')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f'results/noise_rate_{noise_rate}_comparison.png')
        plt.close()
        print(f"Plots saved to results/noise_rate_{noise_rate}_comparison.png")
    except Exception as e:
        print(f"Error plotting results: {e}")
        print("Results have been saved to text file for manual plotting.")


# Function to plot comparison across different noise rates
def plot_noise_rate_comparison(all_results):
    try:
        # Get final test accuracies for each method and noise rate
        noise_rates_list = list(all_results.keys())
        ce_accuracies = [all_results[rate]['CE']['test_accuracies'][-1] for rate in noise_rates_list]
        nce_accuracies = [all_results[rate]['NCE']['test_accuracies'][-1] for rate in noise_rates_list]
        apl_accuracies = [all_results[rate]['APL']['test_accuracies'][-1] for rate in noise_rates_list]

        # Create comparison plot
        plt.figure(figsize=(10, 6))
        plt.plot(noise_rates_list, ce_accuracies, 'bo-', label='CE')
        plt.plot(noise_rates_list, nce_accuracies, 'ro-', label='NCE')
        plt.plot(noise_rates_list, apl_accuracies, 'go-', label='APL (NCE+MAE)')
        plt.xlabel('Noise Rate')
        plt.ylabel('Final Test Accuracy (%)')
        plt.title('Performance Comparison Across Different Noise Rates')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(noise_rates_list)
        plt.savefig('results/noise_rate_comparison.png')
        plt.close()
        print("Noise rate comparison plot saved to results/noise_rate_comparison.png")

        # Save the comparison data to a text file
        with open('results/noise_rate_comparison.txt', 'w') as f:
            f.write("Noise Rate, CE Accuracy, NCE Accuracy, APL Accuracy\n")
            for i, rate in enumerate(noise_rates_list):
                f.write(f"{rate}, {ce_accuracies[i]}, {nce_accuracies[i]}, {apl_accuracies[i]}\n")

    except Exception as e:
        print(f"Error plotting noise rate comparison: {e}")
        print("Results have been saved to text files for manual plotting.")

# Main function to run experiments for all noise rates
def run_all_experiments():
    all_results = {}

    for noise_rate in noise_rates:
        all_results[noise_rate] = run_experiment_for_noise_rate(noise_rate)

    # Plot comparison across different noise rates
    plot_noise_rate_comparison(all_results)

    return all_results


# Run all experiments
if __name__ == "__main__":
    results = run_all_experiments()
