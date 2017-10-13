import json
import codecs
from pprint import pprint

with open('../data/train-v1.1.json') as data_file:
    data = json.load(data_file)
writeDataOnLines = codecs.open('../data/train_lines','w','utf-8')
oldLine = ''
for topic in data["data"]:
    for paragraph in topic["paragraphs"]:
        context = paragraph["context"].replace('\n','').replace('\r','').replace("\t"," ").strip()
        for questionAnswer in paragraph["qas"]:
            question = questionAnswer["question"].replace('\n','').replace('\r','').replace("\t"," ").strip()
            qID = str(questionAnswer["id"]).replace('\n','').replace('\r','').replace("\t"," ").strip()
            for answer in questionAnswer["answers"]:
                answertext = answer["text"].replace('\n','').replace('\r','').replace("\t"," ").strip()
                answerIndex = str(answer["answer_start"]).replace('\n','').replace('\r','').replace("\t"," ").strip()
                answerEnd = str(answer["answer_start"] + len(answertext)).replace('\n','').replace('\r','').replace("\t"," ").strip()
                currentLine = qID + '\t' + context+ '\t' + question + '\t' + answertext + '\t' + str(answerIndex) + '\t' + answerEnd +'\n'
                if currentLine != oldLine:
                    writeDataOnLines.write(currentLine)
                oldLine = currentLine
    writeDataOnLines.flush()
writeDataOnLines.close()
#pprint(data["data"][0]["paragraphs"][1]["context"])
#pprint(data)
print('Done')