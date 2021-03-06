import json
import codecs
from pprint import pprint

trainInput = '../data/train-v1.1.json'
trainOutput = '../data/train_lines'
devInput = '../data/dev-v1.1.json'
devoutput = '../data/dev_lines'

def ConvertJSON(InputPath, OutputPath):
    with open(InputPath) as data_file:
        data = json.load(data_file)
    writeDataOnLines = codecs.open(OutputPath,'w','utf-8')
    oldLine = ''
    counter = 0
    for topic in data["data"]:
        for paragraph in topic["paragraphs"]:
            context = paragraph["context"].replace('\n','').replace('\r','').replace("\t"," ").strip()
            for questionAnswer in paragraph["qas"]:
                question = questionAnswer["question"].replace('\n','').replace('\r','').replace("\t"," ").strip()
                qID = str(questionAnswer["id"]).replace('\n','').replace('\r','').replace("\t"," ").strip()
                # if(qID == '56cf7c634df3c31400b0d840'):
                #     print ("hahahah")
                for answer in questionAnswer["answers"]:
                    answertext = answer["text"].replace('\n','').replace('\r','').replace("\t"," ").strip()
                    answerIndex = str(answer["answer_start"]).replace('\n','').replace('\r','').replace("\t"," ").strip()
                    answerEnd = str(answer["answer_start"] + (len(answertext) - 1)).replace('\n','').replace('\r','').replace("\t"," ").strip()

                    # charIndex = int(answerIndex)
                    # wordIndexStart = convertCharPositionToWordPosition(context,charIndex)
                    #
                    # charIndex = int(answerEnd)
                    # wordIndexEnd = convertCharPositionToWordPosition(context,charIndex)
                    if qID == '5726a975708984140094cd38':
                        print(qID)
                    wordIndexStart, wordIndexEnd = findWordPosition(context, answerIndex, answertext)
                    # print(wordIndexStart)
                    # print(wordIndexEnd)

                    if wordIndexStart < 0:
                        wordIndexStart = 0
                    if wordIndexEnd >= len(context.split()):
                        wordIndexEnd = len(context.split())-1

                    currentLine = qID + '\t' + context+ '\t' + question + '\t' + answertext + '\t' + str(wordIndexStart) + '\t' + str(wordIndexEnd) +'\n'
                    if currentLine != oldLine:
                        writeDataOnLines.write(currentLine)
                    oldLine = currentLine
        writeDataOnLines.flush()
    writeDataOnLines.close()
    print('Done')

def findWordPosition(text, answerIndex, answertext):
    before = text[0:int(answerIndex)]
    if(len(before) > 0 and before[-1] != ' '):
        start = len(before.split())-1
    else:
        start = len(before.split())
    end = start + len(answertext.split()) - 1
    return start, end

def convertCharPositionToWordPosition(text, charPosition):
    allWords = text.split()
    characterCounter = 0
    wordIndex = 0
    for i in range(len(allWords)):
        for j in range(0, len(allWords[i])):
            characterCounter +=1
        characterCounter +=1
        if characterCounter > charPosition:
            if wordIndex >= len(allWords):
                wordIndex = len(allWords) - 1
            return wordIndex
        else:
            wordIndex += 1
    return len(allWords) - 1

ConvertJSON(trainInput, trainOutput)
ConvertJSON(devInput, devoutput)