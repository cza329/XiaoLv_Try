import requests

url = "https://api.zhituapi.com/hs/history/000001.SZ/d/n?token=7903ABE4-F926-4496-B280-B812DA5FD205&st=20240601&et=20250430"

response = requests.get(url)

data = response.json()

print(data)