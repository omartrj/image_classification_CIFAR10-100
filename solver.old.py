import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10
from utils.transform import get_transforms
from net import Net
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np


class Solver:
    def __init__(self, args):
        self.args = args

        # Trasformazioni per il dataset
        pretrained = True
        if args.backbone == "custom":
            pretrained = False
        transform_train, transform_test = get_transforms(args.dataset, pretrained)

        # Dataset e DataLoader
        if args.dataset == "cifar10":
            dataset = CIFAR10
            num_classes = 10
        elif args.dataset == "cifar100":
            dataset = CIFAR100
            num_classes = 100

        self.train_dataset = dataset(
            root=args.data_root, train=True, download=True, transform=transform_train
        )
        self.test_dataset = dataset(
            root=args.data_root, train=False, download=True, transform=transform_test
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=args.batch_size, shuffle=False
        )

        # Modello, ottimizzatore e funzione di perdita
        is_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if is_cuda else "cpu")
        print(f"[INFO] Using device: {self.device}")
        self.model = Net(backbone_name=args.backbone, num_classes=num_classes).to(
            self.device
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Crea la directory per i checkpoint, se non esiste
        os.makedirs(args.checkpoint_path, exist_ok=True)

        # Ripristina il checkpoint, se specificato
        self.start_epoch = 0
        if args.checkpoint_file:
            self.load_checkpoint(args.checkpoint_file)

        # Inizializza il file CSV
        self.csv_file = os.path.join(args.checkpoint_path, "training_log.csv")
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Accuracy", "Test Accuracy"])

        # Parametri per l'early stopping
        self.best_test_accuracy = 0
        self.patience_counter = 0

    def early_stop(self, test_accuracy):
        """Gestisce l'early stopping in base all'accuracy sul test set."""
        if test_accuracy > self.best_test_accuracy:
            self.best_test_accuracy = test_accuracy
            self.patience_counter = 0
            return False  # Continua l'addestramento
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.args.patience:
                print(f"[INFO] Early stopping triggered. Best test accuracy: {self.best_test_accuracy:.2f}%")
                return True  # Interrompe l'addestramento
            return False

    def run(self):
        """Esecuzione del solver, in modalità di addestramento o test"""
        if self.args.test:
            self.test()
        else:
            self.fit()

    def fit(self):
        """Addestramento del modello"""
        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            current_lr = self.optimizer.param_groups[0]["lr"]
            with tqdm(
                total=len(self.train_loader),
                desc=f"Epoch {epoch + 1}/{self.args.epochs}",
                unit="batch",
            ) as pbar:
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    _, predicted = output.max(1)
                    correct += predicted.eq(target).sum().item()
                    total += target.size(0)

                    avg_loss_so_far = total_loss / (batch_idx + 1)
                    pbar.set_postfix(
                        loss=f"{avg_loss_so_far:.4f}", lr=f"{current_lr:.6f}"
                    )
                    pbar.update(1)

            train_accuracy = 100 * correct / total
            test_accuracy = self.test()

            # Salva le metriche nel CSV
            with open(self.csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, train_accuracy, test_accuracy])

            # Controlla l'early stopping
            if self.early_stop(test_accuracy):
                break

            self.scheduler.step(total_loss)

    def save_checkpoint(self, epoch):
        """Salva il checkpoint del modello"""
        checkpoint_path = os.path.join(
            self.args.checkpoint_path, f"model_epoch_{epoch + 1}.pth"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path):
        """Carica il checkpoint del modello"""
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"]
        else:
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} non trovato.")

    def test(self, data_loader=None):
        """Valutazione del modello"""
        self.model.eval()
        correct = 0
        total = 0
        if data_loader is None:
            data_loader = self.test_loader
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = 100 * correct / total
        return accuracy
