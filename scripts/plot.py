import matplotlib.pyplot as plt
import pandas as pd

# Function to plot the training and validation metrics for a given model
def plot_training_metrics(data, model_name):
    epochs = data['epoch']
    train_loss = data['train_loss']
    val_loss = data['test_loss']
    train_acc = data['train_accuracy']
    val_acc = data['test_accuracy']
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss evolution
    axs[0].plot(epochs, train_loss, label='Train Loss', color='blue')
    axs[0].plot(epochs, val_loss, label='Validation Loss', color='orange')
    axs[0].set_title(f'{model_name} - Loss Evolution')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot accuracy evolution
    axs[1].plot(epochs, train_acc, label='Train Accuracy', color='blue')
    axs[1].plot(epochs, val_acc, label='Validation Accuracy', color='orange')
    axs[1].set_title(f'{model_name} - Accuracy Evolution')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Load the uploaded CSV files

    # CIFAR-10
    custom_file_path_cifar10 = './checkpoints/cifar10_custom/training_log.csv'
    resnet_file_path_cifar10 = './checkpoints/cifar10_resnet/training_log.csv'

    # CIFAR-100
    custom_file_path_cifar100 = './checkpoints/cifar100_custom/training_log.csv'
    resnet_file_path_cifar100 = './checkpoints/cifar100_resnet/training_log.csv'

    # Load the data
    custom_data_cifar10 = pd.read_csv(custom_file_path_cifar10)
    resnet_data_cifar10 = pd.read_csv(resnet_file_path_cifar10)
    custom_data_cifar100 = pd.read_csv(custom_file_path_cifar100)
    resnet_data_cifar100 = pd.read_csv(resnet_file_path_cifar100)

    # Generate the plots for each model
    plot_training_metrics(custom_data_cifar10, 'Custom Model on CIFAR-10')
    plot_training_metrics(resnet_data_cifar10, 'ResNet Model on CIFAR-10')
    plot_training_metrics(custom_data_cifar100, 'Custom Model on CIFAR-100')
    plot_training_metrics(resnet_data_cifar100, 'ResNet Model on CIFAR-100')

if __name__ == "__main__":
    main()
