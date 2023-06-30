import json

data = []


str_prompt = "photo of a person, style zwk, closeup"
str_neg = "smiling with teeth(deformed iris, deformed pupils, semi-realistic, cgi, 3d"

data.append('[\n')
data.append('"' + str_prompt + '",\n')
data.append('"' + str_neg + '"\n')
data.append(']\n')



# Convert the list to a string
data_string = ''.join(data)

# Write the string to a JSON file
with open("data.json", "w") as json_file:
    json_file.write(data_string)
