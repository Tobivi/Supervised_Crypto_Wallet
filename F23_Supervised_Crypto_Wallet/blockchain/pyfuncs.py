import requests
from bs4 import BeautifulSoup

def getLastTrans(addr):
    url = f'https://mumbai.polygonscan.com/address/{addr}'# + addr
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

    response = requests.get(url=url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        targ = soup.find(class_='myFnExpandBox_searchVal')

        if targ:
            return targ.text
        else:
            print("Element not found.")
    else:
        print(f"Failed to retrieve the web page. Status code: {response.status_code}")



def MATICtoETH(MATICAMT):
    resp = requests.get('https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest?CMC_PRO_API_KEY=dc0ab239-8691-42c2-823d-b230fc37d488&convert=MATIC')
    data = resp.json()
    MATICperETH = -1

    for entry in data['data']:
        if "symbol" in entry and entry["symbol"] == "ETH":
            MATICperETH = entry['quote']['MATIC']['price']
            break

    if (MATICperETH == -1): return -1
    return str(MATICAMT / MATICperETH)


def parseWebPage(url):
        responseOut = requests.get(url=url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        # responseIn = requests.get(url=urlin, headers=headers)

        # create the soup....thing.....yes
        soup = BeautifulSoup(responseOut.text, 'html.parser')

        # Create a dictionary to store transactions
        transMapOut = {}

        # Find all table rows
        rows = soup.find_all('tr')

        # Iterate through each row
        for row in rows:
            # Extract wallet address and transaction details
            if (not row.find('span', class_='hash-tag')): continue

            walletAddr = ""
            to = ""
            wallet_address_tag = row.find('a', href=lambda x: x and "/address/" in x)

            if wallet_address_tag:
                wallet_address_href = wallet_address_tag['href']
                walletAddr = wallet_address_href.replace("/address/", "")
                to = wallet_address_tag.text.lower().strip()
            else:
                continue

            method = row.find('span', class_='badge').text.strip()
            transaction_details = {}

            if (method.lower() == 'transfer'):
                transaction_details = {
                    'transaction_hash': row.find('a', class_='myFnExpandBox_searchVal').text.strip(),
                    # 'method': method,
                    'block_number': row.find('a', href=True).text.strip(),
                    'timestamp': row.find('td', class_='showDate').text.strip(),
                    'amount': row.find(attrs={"data-bs-title": lambda x: x and "MATIC" in x}).text.strip(),
                    'fee': row.find('td', class_='small text-muted showTxnFee').text.strip(),
                    'gas_price': row.find('td', class_='small text-muted showGasPrice').text.strip(),
                }
            
            elif (to == 'contract creation'):
                # print("contract created")
                continue

            else: continue


            # Check if wallet address is already in the dictionary
            if walletAddr in transMapOut:
                transMapOut[walletAddr].append(transaction_details)
            else:
                transMapOut[walletAddr] = [transaction_details]

        # Print the transactions map
        # for wallet, transactions in transactions_map.items():
        #     print(f"Wallet Address: {wallet}")
        #     for transaction in transactions:
        #         print(f"  Transaction Hash: {transaction['transaction_hash']}")
        #         print(f"  Transaction Type: {transaction['method']}")
        #         print(f"  Block Number: {transaction['block_number']}")
        #         print(f"  Timestamp: {transaction['timestamp']}")
        #         print(f"  Amount: {transaction['amount']}")
        #         print(f"  Fee: {transaction['fee']}")
        #         print(f"  Gas Price: {transaction['gas_price']}")
        #         print("\n")

        return transMapOut


def coalateTransactions(addr):
    urlout = f'https://mumbai.polygonscan.com/txs?a={addr}&f=2'  # only outgoing
    urlin = f'https://mumbai.polygonscan.com/txs?a={addr}&f=3'  # only incoming

    outMap = parseWebPage(urlout)
    inMap = parseWebPage(urlin)

    return inMap, outMap