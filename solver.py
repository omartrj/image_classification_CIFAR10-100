import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10
from utils.transform import get_transforms
from utils.logger import get_logger
import logging
from net import Net
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class Solver:
    def __init__(self, args):
        self.args = args
        self.logger = get_logger()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {'CUDA' if self.device.type == 'cuda' else 'CPU'}")

        # Determina se usare trasformazioni pre-addestrate
        pretrained = self.args.backbone != "custom"
        transform_train, transform_test = get_transforms(self.args.dataset, pretrained)

        # Determina il dataset e il numero di classi in base al tipo di dataset selezionato
        dataset_class = CIFAR10 if self.args.dataset == "cifar10" else CIFAR100
        num_classes = 10 if self.args.dataset == "cifar10" else 100

        # Crea i DataLoader per il training e il test
        self.train_dataset = dataset_class(
            root=self.args.data_root,
            train=True,
            download=True,
            transform=transform_train,
        )
        self.test_dataset = dataset_class(
            root=self.args.data_root,
            train=False,
            download=True,
            transform=transform_test,
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.args.batch_size, shuffle=False
        )

        # Inizializza il modello, la funzione di loss e l'ottimizzatore
        self.model = Net(backbone_name=self.args.backbone, num_classes=num_classes).to(
            self.device
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=3
        )

        # Crea la directory per salvare i checkpoint e carica un checkpoint se specificato
        os.makedirs(self.args.checkpoint_path, exist_ok=True)
        self.start_epoch = 0
        if self.args.checkpoint_file:
            self._load_checkpoint(self.args.checkpoint_file)

        # Imposta i parametri per l'early stopping
        self.best_test_accuracy = 0
        self.patience_counter = 0

    def _load_checkpoint(self, checkpoint_path):
        # Carica un checkpoint salvato per riprendere l'addestramento
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False) # weights_only=False per evitare warning
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"]
        else:
            raise FileNotFoundError(
                f"[ERROR] Checkpoint file not found: {checkpoint_path}"
            )

    def _save_checkpoint(self, epoch):
        # Salva un checkpoint del modello
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

    def run(self):
        # Avvia la modalità test o addestramento a seconda dei parametri
        if self.args.test:
            self.test()
        else:
            self.csv_file = os.path.join(self.args.checkpoint_path, "training_log.csv")
            self._init_training_log()
            self.fit()

    def _init_training_log(self):
        # Inizializza un file CSV per registrare le metriche durante l'addestramento
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_accuracy", "test_accuracy", "train_loss", "test_loss"]
            )

    def fit(self):
        for epoch in range(self.start_epoch, self.args.epochs):

            # Esegue un'epoca di addestramento
            self._train_one_epoch(epoch)

            # Salva un checkpoint del modello
            self._save_checkpoint(epoch)

            # Valuta il modello sul training set e sul test set
            train_accuracy, train_loss = self._evaluate(self.train_loader)
            test_accuracy, test_loss = self._evaluate(self.test_loader)

            # Registra le metriche nel file CSV
            self._log_metrics(
                epoch, train_accuracy, test_accuracy, train_loss, test_loss
            )

            # Controlla i criteri di early stopping
            if self._early_stop(test_accuracy, patience=5):
                print(
                    f"[INFO] Early stopping activated at epoch {epoch + 1}. Best test accuracy: {self.best_test_accuracy:.2f}%"
                )
                break

            # Aggiorna il learning rate utilizzando lo scheduler
            self.scheduler.step(test_loss)

    def _train_one_epoch(self, epoch):
        # Addestra il modello per una singola epoca
        self.model.train()

        with tqdm(
            total=len(self.train_loader),
            desc=f"Epoch {epoch + 1}/{self.args.epochs}",
            unit="batch",
        ) as pbar:
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Azzeramento dei gradienti, forward pass, calcolo della loss e backward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                # Aggiorna la barra di avanzamento
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=self.optimizer.param_groups[0]["lr"],
                )
                pbar.update(1)

    def _evaluate(self, data_loader):
        # Valuta il modello su un data loader specifico
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        # Configura il progress bar con uno spinner e una barra, con stile personalizzato
        with Progress(
            SpinnerColumn(),
            TextColumn(
                f"Evaluating on {'test' if data_loader == self.test_loader else 'train'} set"
            ),
            transient=True,  # Rimuove la barra dopo il completamento
        ) as progress:
            task = progress.add_task("eval", total=len(data_loader))

            with torch.no_grad():
                for data, target in data_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    total_loss += self.criterion(output, target).item()
                    correct += output.argmax(dim=1).eq(target).sum().item()
                    total += target.size(0)

                    # Avanza la barra di completamento
                    progress.update(task, advance=1)

        # Calcola e restituisce le metriche
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(data_loader)
        return accuracy, avg_loss

    def _log_metrics(self, epoch, train_accuracy, test_accuracy, train_loss, test_loss):
        # Registra le metriche di addestramento e test in un file CSV
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch + 1, train_accuracy, test_accuracy, train_loss, test_loss]
            )

    def _early_stop(self, test_accuracy, patience):
        # Controlla i criteri di early stopping basati sull'accuracy del test
        if test_accuracy > self.best_test_accuracy:
            self.best_test_accuracy = test_accuracy
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                return True
            return False

    def test(self):
        # Valuta il modello sul test set e stampa i risultati
        accuracy, loss = self._evaluate(self.test_loader)
        print(f"[INFO] Test Accuracy: {accuracy:.2f}% | Test Loss: {loss:.4f}")

        conf_matrix, report = self._generate_test_report()
        # Salva i risultati su file
        np.save(
            os.path.join(self.args.checkpoint_path, "confusion_matrix.npy"), conf_matrix
        )
        with open(
            os.path.join(self.args.checkpoint_path, "classification_report.json"), "w"
        ) as f:
            json.dump(report, f, indent=4)

        print(f"[INFO] Confusion matrix and metrics report saved.")

    def _generate_test_report(self):
        # Genera e salva la matrice di confusione e il report di classificazione
        all_targets, all_predictions = [], []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(output.argmax(dim=1).cpu().numpy())

        conf_matrix = confusion_matrix(all_targets, all_predictions)
        report = classification_report(all_targets, all_predictions, output_dict=True)

        return conf_matrix, report
