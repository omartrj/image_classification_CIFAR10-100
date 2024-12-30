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
    print("[INFO] Model: ", config['backbone'])
    print("[INFO] Dataset: ", config['dataset'])

    subprocess.run(command)

    print("[INFO] Done\n")

    