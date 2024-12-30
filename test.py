import os
import json
from utils.helper import run_experiment


def main():

    # Configurazioni
    configs = [
        "./configs/cifar10_custom.json",
        #"./configs/cifar100_custom.json",
        "./configs/cifar10_resnet.json",
        #"./configs/cifar100_resnet.json",
    ]

    # Esegui gli esperimenti in modalità test
    for config_file in configs:
        with open(config_file, "r") as f:
            config = json.load(f)

        # Seleziona il modello da testare (l'ultimo epoch)
        config["checkpoint_file"] = os.path.join(
            config["checkpoint_path"], f"model_epoch_{config['epochs']}.pth"
        )

        run_experiment(config, test=True)


if __name__ == "__main__":
    main()
