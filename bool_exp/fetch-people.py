import requests

file = open('people-names.txt', 'r')
names = file.readlines()
for name in names:
    name = name.strip()
    response = requests.get('https://en.wikipedia.org/w/api.php', params={'action': 'query', 'format': 'json', 'titles': name, 'prop': 'extracts', 'explaintext': True, 'exlimit': 'max', }).json()
    page = next(iter(response['query']['pages'].values()))
    content = (page['extract'])
    file_title = name.lower().replace(" ", "_")

    file_to_save = open("people/" + file_title + ".txt", "wb")
    file_to_save.write(content.encode("utf-8"))
    file_to_save.close()
file.close()
