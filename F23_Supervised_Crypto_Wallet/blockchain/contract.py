from typing import Any
from web3 import Web3, HTTPProvider
from web3.types import TxParams
from web3.contract.contract import Contract
from json import load

w3 = Web3(HTTPProvider("https://sepolia.infura.io/v3/0776cf37dfb04efdacd478388c7c1dec"))

parse_json_file = lambda filepath: load(open(filepath))

w3confs: dict[str, Any] = parse_json_file("./secrets/blockchain.json")
private_key: str = w3confs["private_key"]
account: str = w3confs["account"]
contract: Contract = w3.eth.contract(w3confs["contract"], abi=parse_json_file("./abi.json"))

def transact(func_name: str, txparams: TxParams):
	tx = contract.functions.__getattr__(func_name)().build_transaction(txparams)
	signed_tx = w3.eth.account.sign_transaction(tx, private_key)
	return w3.eth.send_raw_transaction(signed_tx.rawTransaction)

transact("deposit", {
	"from": account,
	"value": w3.to_wei(50, "wei"),
	"nonce": w3.eth.get_transaction_count(account) # type: ignore
})
