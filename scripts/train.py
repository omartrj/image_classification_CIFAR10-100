import json
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

    # Esegui gli esperimenti in modalità train
    for config_file in configs:
        with open(config_file, "r") as f:
            config = json.load(f)

        run_experiment(config, test=False)


if __name__ == "__main__":
    main()
