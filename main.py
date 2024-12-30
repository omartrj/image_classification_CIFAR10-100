import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser()

    # Configurazioni di base
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Numero di epoche")
    parser.add_argument(
        "--backbone",
        type=str,
        default="custom",
        choices=["resnet50", "vgg16", "custom"],
        help="Backbone da utilizzare",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar10", "cifar100"],
        help="Dataset da utilizzare",
    )
    parser.add_argument(
        "--data_root", type=str, default="./data", help="Percorso del dataset"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints",
        help="Percorso dei checkpoint",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default=None,
        help="Percorso del file di checkpoint da caricare",
    )
    parser.add_argument(
        "--test", action="store_true", help="Valuta il modello sul test set"
    )

    args = parser.parse_args()

    solver = Solver(args)
    solver.run()

if __name__ == "__main__":
    main()
