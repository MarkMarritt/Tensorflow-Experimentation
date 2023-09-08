import pandas as pd
import numpy as np
import random
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import  pad_sequences

#params
maxLength = 50
wordBank = 15000 # i believe should be the same as num_words
denseLayerSize = 16
label1 = "RACECOMM"
label2 = "RACERATE"
epochNum = 30
testFrac = 0.8

data = pd.read_csv('C:\Code\Bet turtle\Data tables\HorseCommentsSample.csv')
raceData = data[["RACECOMM", "RACERATE"]]
invalidData = raceData[raceData["RACECOMM"] == 0].index # remove any data with empty racerate 
raceData.drop(invalidData)
testNumber = int(testFrac * len(raceData))

raceData.replace([1,2],0, inplace=True) # convert all the data into 1 or 0 (above or below 2.5, so it is now binary)
raceData.replace([3,4,5],1, inplace=True)


raceDataTrain = raceData[0:testNumber]#split the data into test and train data
raceDataTest = raceData[testNumber:]

def padData(dataframe, label1 = label1, label2 = label2, maxLength = maxLength):
    """takes a pandas dataframe, labels need to correspond to the dataframes labels e.g. RACECOMM. Outputs padded data and numbers converted to tokens"""
    data = dataframe[label1]
    token = Tokenizer(wordBank, oov_token="<00V>")
    token.fit_on_texts(data) #this works, converts all the words to numbers
    tokennedText = np.ndarray.tolist(pad_sequences(token.texts_to_sequences(data), padding="post",maxlen=maxLength ))
    tokennedData = pd.DataFrame({"TEXT": tokennedText,
                                "VALUE": dataframe[label2]}) #combines back into a pandas dataframe
    return tokennedData

padDataTrain = padData(raceDataTrain)
padDataTest = padData(raceDataTest)
#now have 2 sets of padded data
dfADD = pd.DataFrame()
labels = []
texts = []

for j in range(10):
    for i in range(len(padDataTrain)):
        label = padDataTrain["VALUE"][i]
        text = padDataTrain["TEXT"][i]
        random.shuffle(text)
        labels.append(label)
        texts.append(text)
dfADD["VALUE"] = labels
dfADD["TEXT"] = texts
padDataTrain = pd.concat([padDataTrain, dfADD])


trainingText = np.asarray(padDataTrain["TEXT"].tolist())
trainingValue = np.asarray(padDataTrain["VALUE"].tolist())
testText = np.asarray(padDataTest["TEXT"].tolist())
testValue = np.asarray(padDataTest["VALUE"].tolist())


################################################################

model = keras.Sequential()
model.add(keras.layers.Embedding(wordBank +1, 500 ))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
optimizer = keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics="accuracy")

model.fit(trainingText, trainingValue, epochs = 20, batch_size=50, validation_data=(testText,testValue))

model.save("C:\Code\Bet turtle\ModelSaves\VersionThree")

