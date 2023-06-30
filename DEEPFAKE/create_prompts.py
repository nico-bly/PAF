import json


# Define the file path
file_path = "prompts/prompts_SD_Humans_V4.json"

Ethinicity= ["South Asian", "European", "African", "East Asian", "Native American", "scandinavian", "Hispanic", "Middle Eastern"]
Gender=["Man", "Woman"]
Age = ["","old"]
Glasses = ["",", with glasses,"]
Beard =["", ", with beard,"]
data = []
data.append( "{ \n  " + '"prompts trains"' + ":[ \n")
count =0

for E in Ethinicity:
    for A in Age:
        for g in range(2):
            for b in range(2):
                if Gender[g] == "Man":
                    data.append('[ \n')
                    str_prompt ="photo of a person, style zwk, closeup, {} {} with {} ethnicity {} {}".format(A,Gender[g],E,Beard[b],Glasses[g] )
                    str_neg =  " cgi {} {} {} (deformed iris, deformed pupils, semi-realistic, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), wrong anatomy, mutation, amputation".format(Beard[(b+1)%2],Glasses[(g+1)%2],Gender[(g+1)%2])
                    data.append('"'+ str_prompt + '"')
                    data.append(',\n')
                    data.append('"' +str_neg+ '"')
                    data.append("],"+"\n")
                    count=count+1
                    data.append('[ \n')
                    str_prompt ="photo of a person, style zwk, closeup, bald, {} {} with {} ethnicity {} {}".format(A,Gender[g],E,Beard[b],Glasses[g] )
                    str_neg =  " hair, cgi {} {} {} (deformed iris, deformed pupils, semi-realistic, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), wrong anatomy, mutation, amputation".format(Beard[(b+1)%2],Glasses[(g+1)%2],Gender[(g+1)%2])
                    data.append('"'+ str_prompt + '"')
                    data.append(',\n')
                    data.append('"' +str_neg+ '"')
                    data.append("],"+"\n")
                    count=count+1
                else:
                    data.append("[ \n")
                    str_prompt = "photo of a person, style zwk, closeup, {} {} with {} ethnicity {}".format(A,Gender[g],E,Glasses[g])
                    str_neg =  "smiling with teeth, short hair, {} {} {} (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), wrong anatomy, mutation, amputation".format(Beard[1],Glasses[(g+1)%2],Gender[(g+1)%2])
                    data.append('"'+ str_prompt + '" ')
                    data.append(',\n')
                    data.append('"' +str_neg+ '"')
                    data.append("],"+"\n")
                    count=count+1
                    data.append("[ \n")
                    str_prompt = "photo of a person, style zwk, closeup, {} {} with {} ethnicity {}".format(A,Gender[g],E,Glasses[g])
                    str_neg =  "smiling with teeth, long hair, {} {} {} (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), wrong anatomy, mutation, amputation".format(Beard[1],Glasses[(g+1)%2],Gender[(g+1)%2])
                    data.append('"'+ str_prompt + '" ')
                    data.append(',\n')
                    data.append('"' +str_neg+ '"')
                    data.append("],"+"\n")
                    count=count+1
for E in Ethinicity:
    for g in range(2):
            if Gender[g] == "Man":
                data.append('[ \n')
                str_prompt ="photo of a person, style zwk, closeup, young boy with {} ethnicity {} ".format(E,Glasses[g] )
                str_neg =  "cgi {} {} {}, (deformed iris, deformed pupils, semi-realistic, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), wrong anatomy, mutation, amputation".format(Beard[(b+1)%2],Glasses[(g+1)%2],Gender[(g+1)%2])
                data.append('"'+ str_prompt + '"')
                data.append(',\n')
                data.append('"' +str_neg+ '"')                                           
                data.append("],"+"\n")
                count=count+1
            else:
                data.append("[ \n")
                str_prompt = "photo of a person, style zwk, closeup, young girl with {} ethnicity {}".format(E,Glasses[g])
                str_neg =  "smiling with teeth, {} {} {}, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), wrong anatomy, mutation, amputation".format(Beard[1],Glasses[(g+1)%2],Gender[(g+1)%2])
                data.append('"'+ str_prompt + '" ')
                data.append(',\n')
                data.append('"' +str_neg+ '"')
                data.append("],"+"\n")
                count=count+1



data.append("] \n }")
print(count)

data_string = ''.join(data)
# Write the dictionary to a JSON file
with open(file_path, "w") as json_file:
    #json.dump(data, json_file, indent="\n")
    json_file.write(data_string)