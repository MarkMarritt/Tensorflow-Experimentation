import pandas as pd
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import  pad_sequences

#params
maxLength = 100
EmbeddingSize = 10000 # i believe should be the same as num_words
denseLayerSize = 16
testNumber = 750
label1 = "RACECOMM"
label2 = "RACERATE"
epochNum = 30

data = pd.read_csv('C:\Code\Bet turtle\Data tables\HorseCommentsSample.csv')
raceData = data[["RACECOMM", "RACERATE"]]
raceDataTrain = raceData[0:testNumber]#split the data into test and train data
raceDataTest = raceData[testNumber:]

def padData(dataframe, label1 = label1, label2 = label2, maxLength = maxLength):
    """takes a pandas dataframe, labels need to correspond to the dataframes labels e.g. RACECOMM. Outputs padded data and numbers converted to tokens"""
    data = dataframe[label1]
    token = Tokenizer(num_words = 1000, oov_token="<00V>")
    token.fit_on_texts(data) #this works, converts all the words to numbers
    tokennedText = np.ndarray.tolist(pad_sequences(token.texts_to_sequences(data), padding="post",maxlen=maxLength ))
    tokennedData = pd.DataFrame({"TEXT": tokennedText,
                                "VALUE": dataframe[label2]}) #combines back into a pandas dataframe
    return tokennedData

padDataTrain = padData(raceDataTrain)
padDataTest = padData(raceDataTest)
#now have 2 sets of padded data

trainingText = np.asarray(padDataTrain["TEXT"].tolist())
trainingValue = np.asarray(padDataTrain["VALUE"].tolist())
testText = np.asarray(padDataTest["TEXT"].tolist())
testValue = np.asarray(padDataTest["VALUE"].tolist())

"""#here is just a simple model
model = keras.Sequential() # make a model object
model.add(keras.layers.Embedding(1000, EmbeddingSize,input_length=maxLength)) #embedding size is the number of neurons of this layer I believe
model.add(keras.layers.Normalization())
#model.add(keras.layers.GlobalAveragePooling1D()) # not too sure what thia does, need to look into
model.add(keras.layers.Dense(denseLayerSize, activation="relu")) # this layer biases linearly based on the training data
model.add(keras.layers.Dense(5, activation="softmax")) # this is the output layer, hence it has 1 neuron, which should hopefully be the value.
model.compile(loss= "sparse_categorical_crossentropy", optimizer = "adam", metrics = "accuracy")
model.summary()





result = model.fit(trainingText, trainingValue, epochs = epochNum, validation_data = (testText, testValue))

model.save("C:\Code\Bet turtle\ModelSaves\VersionOne")
model.evaluate(testText, testValue, verbose=2)"""