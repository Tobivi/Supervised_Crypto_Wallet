# F23_Supervised_Crypto_Wallet
Get help from AI in managing your (or you child's) cryptocurrencies, with measures in place to ensure a safe introduction into the area.

## Features
* ask AI questions through our easy-to-use front end
* put spending limits on accounts
* be notified about any excessive spending
* has an easy-to-use interface

## Running Locally
1. clone the project
2. create a folder called "secrets" in your root directory with a file called `config.json` that has the following in it:
```json
{
  "APITOKEN": "chatGPT_token",
  "bestHidden": 64
}
```
3. run `pip install -r requirements.txt`
4. run `python webserver.py`


## Errors and Debugging
* if getting an error in the python console about incorrect model dimensions, delete the `model.pth` file and re-run the program
* if the web-page keeps popping up an alert that says "ERROR" and you want a more detailed explanation, it will be in the console

## Notes
* the `bestHidden` field in the JSON file is ONLY necessary if you are using the model currently in the github, and will be automatically changed by the program when re-training or making a new model
* get your chatGPT token [here](https://platform.openai.com/api-keys)
