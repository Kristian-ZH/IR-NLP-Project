import requests

file = open('data-generation/people-names.txt', 'r')
names = file.readlines()
for name in names:
    name = name.strip()
    response = requests.get('https://en.wikipedia.org/w/api.php',params={'action': 'query','format': 'json','titles': name,'prop': 'extracts','explaintext': True,'exlimit': 'max',}).json()
    page = next(iter(response['query']['pages'].values()))
    content=(page['extract'])
    file_title=name.lower().replace(" ", "_")

    file_to_save = open("data-generation/people/" + file_title + ".txt", "w")
    file_to_save.write(content)
    file_to_save.close()
file.close()