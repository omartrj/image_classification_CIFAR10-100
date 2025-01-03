import os
import json
import csv
from utils.helper import run_experiment


def main():

    # Configurazioni
    configs = [
        "./configs/cifar10_custom.json",
        "./configs/cifar100_custom.json",
        "./configs/cifar10_resnet.json",
        "./configs/cifar100_resnet.json",
        "./configs/cifar10_vgg.json",
        "./configs/cifar100_vgg.json",
    ]

    # Esegui gli esperimenti in modalità test
    for config_file in configs:
        with open(config_file, "r") as f:
            config = json.load(f)

        # Seleziona il modello da testare (l'ultimo epoch nel file training_log.csv)
        log_file = os.path.join(config["checkpoint_path"], "training_log.csv")
        reader = csv.DictReader(open(log_file, "r"))
        last_row = list(reader)[-1]
        last_epoch = int(last_row["epoch"])
        config["checkpoint_file"] = os.path.join(
            config["checkpoint_path"], f"model_epoch_{last_epoch}.pth"
        )

        run_experiment(config, test=True)


if __name__ == "__main__":
    main()
