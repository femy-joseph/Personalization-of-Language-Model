#######################################Code to create the json file as needed by pyterrier###############
import json
with open("/content/sample_data/Lamp4.json","r") as file:
    data=json.load(file)
#json.dumps(data[0]['profile'])

with open("profile.json", "w") as f:
    json.dump(data[0]['profile'], f)

with open("/content/profile.json","r") as file:
    profiles=json.load(file)
profile_texts = []
for i in range(len(profiles)):
  profile_dict = {}
  profile_dict['docno'] = 'doc' + str(i + 1)
  profile_dict['text'] = profiles[i]['text']
  profile_texts.append(profile_dict)

#print(profile_texts)
with open("/content/sample_data/profile_texts.json", "w") as f:
    json.dump(profile_texts, f)