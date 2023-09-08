import pandas as pd
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import  pad_sequences

maxLength = 50
wordBank = 10000 # i believe should be the same as num_words
denseLayerSize = 16
testNumber = 500
label1 = "RACECOMM"
label2 = "RACERATE"
epochNum = 30

data = pd.read_csv('C:\Code\Bet turtle\Data tables\HorseCommentsSample.csv')
raceData = data[["RACECOMM", "RACERATE"]]
raceDataTrain = raceData[0:testNumber]#split the data into test and train data
raceDataTest = raceData[testNumber:]

token = Tokenizer(wordBank, oov_token="<00V>")

def padData(dataframe, label1 = label1, label2 = label2, maxLength = maxLength):
    """takes a pandas dataframe, labels need to correspond to the dataframes labels e.g. RACECOMM. Outputs padded data and numbers converted to tokens"""
    data = dataframe[label1]
    
    token.fit_on_texts(data) #this works, converts all the words to numbers
    tokennedText = np.ndarray.tolist(pad_sequences(token.texts_to_sequences(data), padding="post",maxlen=maxLength ))
    tokennedData = pd.DataFrame({"TEXT": tokennedText,
                                "VALUE": dataframe[label2]}) #combines back into a pandas dataframe
    return tokennedData


def dummyData(text, label):
    token.fit_on_texts([text]) 
    tokennedText = np.ndarray.tolist(pad_sequences(token.texts_to_sequences([text]), padding="post",maxlen=maxLength ))
    return tokennedText, label


padDataTrain = padData(raceDataTrain)
padDataTest = padData(raceDataTest)
#now have 2 sets of padded data

trainingText = np.asarray(padDataTrain["TEXT"].tolist())
trainingValue = np.asarray(padDataTrain["VALUE"].tolist())/4
testText = np.asarray(padDataTest["TEXT"].tolist())
testValue = np.asarray(padDataTest["VALUE"].tolist())/4

newModel = keras.models.load_model("C:\Code\Bet turtle\ModelSaves\VersionTwo")
newModel.evaluate(testText, testValue, verbose = 2)

dummyText1 = dummyData("A fair handicap, but again they finished well spread out. While this not form to take too literally, it does look sensible.", 3)
dummyText2 = dummyData("A fair little handicap and it was dominated throughout by the market principals. The form makes sense at face value but the first two could be better than they were able to show, as the pace wasn't strong.", 3)
dummyText3 = dummyData("Previous form was thin on the ground in this fillies' maiden race which was run at just a steady pace to past halfway. The two dead-heaters were one-two throughout.", 1)
dummyText4 = dummyData("Excellent, best race ive seen in a while.", 4)
result1 = newModel.predict(dummyText1[0])
print(np.mean(result1))
result2 = newModel.predict(dummyText2[0])
print(np.mean(result2))
result3 = newModel.predict(dummyText3[0])
print(np.mean(result3))
result4 = newModel.predict(dummyText4[0])
print(np.mean(result4))