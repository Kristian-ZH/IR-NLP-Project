import requests
import os
import glob

def generateTexts(path):
    file = open(path, 'r')
    names = file.readlines()
    dir_name = os.path.abspath(os.path.join(__file__, '..', 'people'))


    # Clear existing files in the dir
    # existing_files = glob.glob(dir_name+"*")
    # for f in existing_files:
    #     os.remove(f)

    for name in names:
        name = name.strip()
        response = requests.get('https://en.wikipedia.org/w/api.php',params={'action': 'query','format': 'json','titles': name,'prop': 'extracts','explaintext': True,'exlimit': 'max',}).json()
        page = next(iter(response['query']['pages'].values()))
        content=(page['extract'])
        file_title=name.lower().replace(" ", "_")

        file_to_save = open(dir_name + "/" + file_title + ".txt", "w+")
        file_to_save.write(content)
        file_to_save.close()
    file.close()
