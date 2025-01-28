import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os


def save_figure(fig, directory, filename):
    """Save the figure in the specified directory."""
    os.makedirs(directory, exist_ok=True)
    fig_path = os.path.join(directory, filename)
    fig.savefig(fig_path)
    plt.close(fig)


# Function to plot the training and validation metrics for a given model
def plot_training_metrics(data, model_name, output_dir):
    epochs = data["epoch"]
    train_loss = data["train_loss"]
    val_loss = data["test_loss"]
    train_acc = data["train_accuracy"]
    val_acc = data["test_accuracy"]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot loss evolution
    axs[0].plot(epochs, train_loss, label="Train Loss", color="blue")
    axs[0].plot(epochs, val_loss, label="Validation Loss", color="orange")
    axs[0].set_title(f"{model_name} - Loss Evolution")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # Plot accuracy evolution
    axs[1].plot(epochs, train_acc, label="Train Accuracy", color="blue")
    axs[1].plot(epochs, val_acc, label="Validation Accuracy", color="orange")
    axs[1].set_title(f"{model_name} - Accuracy Evolution")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    save_figure(fig, output_dir, f"{model_name}_metrics.png")


# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, model_name, output_dir, class_names=None):
    if conf_matrix.shape[0] > 20:  # Simplify for large matrices like CIFAR-100
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig, ax = plt.subplots(figsize=(8, 8))

    cax = ax.matshow(conf_matrix, cmap="coolwarm")
    fig.colorbar(cax)

    # Add values inside the matrix for smaller matrices
    if conf_matrix.shape[0] <= 20:
        for (i, j), val in np.ndenumerate(conf_matrix):
            ax.text(j, i, f"{val}", ha="center", va="center", color="black")

    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # Add class names for CIFAR-10
    if class_names and len(class_names) == 10:  # Check for CIFAR-10
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, ha="center")
        ax.set_yticklabels(class_names)

    save_figure(fig, output_dir, f"{model_name}_confusion_matrix.png")


def save_most_confused_classes(conf_matrix, model_name, output_dir, class_names):
    # Check if the number of class names matches the size of the confusion matrix
    if len(class_names) != conf_matrix.shape[0]:
        raise ValueError(
            f"Mismatch between number of classes ({len(class_names)}) and confusion matrix size ({conf_matrix.shape[0]})."
        )

    confused_pairs = []
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if i != j and conf_matrix[i, j] > 0:
                confused_pairs.append(
                    (class_names[i], class_names[j], conf_matrix[i, j])
                )

    confused_pairs = sorted(confused_pairs, key=lambda x: x[2], reverse=True)
    most_confused_path = os.path.join(
        output_dir, f"{model_name}_most_confused_classes.txt"
    )
    with open(most_confused_path, "w") as f:
        for cls1, cls2, count in confused_pairs[:10]:  # Save top 10 most confused pairs
            f.write(f"{cls1} <-> {cls2}: {count} times\n")


def main():
    # File paths for CIFAR-10 and CIFAR-100
    datasets = {
        "CIFAR-10": [
            ("./checkpoints/cifar10_custom/", 10),
            ("./checkpoints/cifar10_resnet/", 10),
        ],
        "CIFAR-100": [
            ("./checkpoints/cifar100_custom/", 100),
            ("./checkpoints/cifar100_resnet/", 100),
        ],
    }

    # Class names for CIFAR-10 and CIFAR-100
    cifar10_classes = [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]

    cifar100_classes = [
        "Apple",
        "Aquarium fish",
        "Baby",
        "Bear",
        "Beaver",
        "Bed",
        "Bee",
        "Beetle",
        "Bicycle",
        "Bottle",
        "Bowl",
        "Boy",
        "Bridge",
        "Bus",
        "Butterfly",
        "Camel",
        "Can",
        "Castle",
        "Caterpillar",
        "Cattle",
        "Chair",
        "Chimpanzee",
        "Clock",
        "Cloud",
        "Cockroach",
        "Couch",
        "Crab",
        "Crocodile",
        "Cup",
        "Dinosaur",
        "Dolphin",
        "Elephant",
        "Flatfish",
        "Forest",
        "Fox",
        "Girl",
        "Hamster",
        "House",
        "Kangaroo",
        "Keyboard",
        "Lamp",
        "Lawn mower",
        "Leopard",
        "Lion",
        "Lizard",
        "Lobster",
        "Man",
        "Maple tree",
        "Motorcycle",
        "Mountain",
        "Mouse",
        "Mushroom",
        "Oak tree",
        "Orange",
        "Orchid",
        "Otter",
        "Palm tree",
        "Pear",
        "Pickup truck",
        "Pine tree",
        "Plain",
        "Plate",
        "Poppy",
        "Porcupine",
        "Possum",
        "Rabbit",
        "Raccoon",
        "Ray",
        "Road",
        "Rocket",
        "Rose",
        "Sea",
        "Seal",
        "Shark",
        "Shrew",
        "Skunk",
        "Skyscraper",
        "Snail",
        "Snake",
        "Spider",
        "Squirrel",
        "Streetcar",
        "Sunflower",
        "Sweet pepper",
        "Table",
        "Tank",
        "Telephone",
        "Television",
        "Tiger",
        "Tractor",
        "Train",
        "Trout",
        "Tulip",
        "Turtle",
        "Wardrobe",
        "Whale",
        "Willow tree",
        "Wolf",
        "Woman",
        "Worm",
    ]

    for dataset, paths in datasets.items():
        for path, num_classes in paths:
            model_name = path.split("/")[-2]
            output_dir = os.path.join(path, "figures")

            # Load training logs
            training_log_path = os.path.join(path, "training_log.csv")
            training_data = pd.read_csv(training_log_path)
            plot_training_metrics(training_data, model_name, output_dir)

            # Load confusion matrix
            conf_matrix_path = os.path.join(path, "confusion_matrix.npy")
            conf_matrix = np.load(conf_matrix_path)

            # Select correct class names based on dataset
            class_names = cifar10_classes if num_classes == 10 else cifar100_classes

            # Plot confusion matrix and save most confused classes
            if len(class_names) == 10:
                plot_confusion_matrix(conf_matrix, model_name, output_dir, class_names)
            else:
                plot_confusion_matrix(conf_matrix, model_name, output_dir)
            save_most_confused_classes(conf_matrix, model_name, output_dir, class_names)


if __name__ == "__main__":
    main()
