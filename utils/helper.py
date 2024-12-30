import subprocess
import os

def run_experiment(config, test=False):
    command = [
        "python", "main.py",
        f"--lr={config['lr']}",
        f"--batch_size={config['batch_size']}",
        f"--epochs={config['epochs']}",
        f"--backbone={config['backbone']}",
        f"--dataset={config['dataset']}",
        f"--data_root={config['data_root']}",
        f"--checkpoint_path={config['checkpoint_path']}",
    ]

    if config.get("checkpoint_file"):
        command.append(f"--checkpoint_file={config['checkpoint_file']}")

    if test:
        command.append("--test")


    print(f"[INFO] Running command: {' '.join(command)}")
    print("[INFO] Model: ", "Custom" if config['backbone'] == "custom" else "ResNet50")
    print("[INFO] Dataset: ", "CIFAR10" if config['dataset'] == "cifar10" else "CIFAR100")

    subprocess.run(command)

    print("[INFO] Done\n")

    