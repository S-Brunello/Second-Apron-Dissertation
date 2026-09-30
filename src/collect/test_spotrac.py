import requests

url = "https://www.spotrac.com/nba/cap/_/year/2022"

response = requests.get(url)
print("Status code:", response.status_code)
print("Length:", len(response.text))
print(response.text[:500])
