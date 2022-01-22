import requests
import os
import glob

corpus_path = "corpus/combined.txt"

file = open(corpus_path, 'r')
names = file.readlines()
dir_name = "DataGeneration/people"


# Clear existing files in the dir
# existing_files = glob.glob(dir_name+"*")
# for f in existing_files:
#     os.remove(f)

for name in names:
    name = name.strip()
    response = requests.get('https://en.wikipedia.org/w/api.php',params={'action': 'query','format': 'json','titles': name,'prop': 'extracts','explaintext': True,'exlimit': 'max',}).json()
    page = next(iter(response['query']['pages'].values()))
    content=(page['extract'])
    file_title=name.replace(" ", "_")

    file_to_save = open(dir_name + "/" + file_title, "w+")
    file_to_save.write(content)
    file_to_save.close()
file.close()
