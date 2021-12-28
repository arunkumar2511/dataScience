import requests

xv = requests.get("https://raw.githubusercontent.com/vzhou842/profanity-check/master/profanity_check/data/clean_data.csv")
open('dataset.csv', 'wb').write(xv.content)
