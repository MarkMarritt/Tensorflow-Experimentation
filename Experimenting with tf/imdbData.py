import pandas as pd
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


data = pd.read_csv("C:\Code\Bet turtle\Data tables\IMDB Dataset.csv")
data = data[0:20000]
data.replace("positive", 1, inplace=True)
data.replace("negative", 0, inplace=True)

review = data["review"]
sentiment = data["sentiment"]
token = Tokenizer(num_words = 10000, oov_token="<00v>")
token.fit_on_texts(review)
review = np.ndarray.tolist(pad_sequences(token.texts_to_sequences(review), padding="post", maxlen= 200))

newData = pd.DataFrame({"REVIEW" : review, 
                        "SENTIMENT" : sentiment})

testFrac = 0.8
cutoff = int(testFrac* len(newData))

train = newData[0:cutoff]
test = newData[cutoff:]
trainText = train["REVIEW"].tolist()
trainLabel = train["SENTIMENT"].tolist()
testText = test["REVIEW"].tolist()
testLabel = test["SENTIMENT"].tolist()

model = keras.Sequential()
model.add(keras.layers.Embedding(10001, 500, input_length = 200))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation= "sigmoid"))
optimsier = keras.optimizers.Adam(learning_rate=0.00005)
model.compile(optimizer=optimsier, loss= "binary_crossentropy", metrics="accuracy")

model.fit(trainText, trainLabel, epochs=50, batch_size=100, validation_data=(testText,testLabel))

model.save("C:\Code\Bet turtle\ModelSaves\IMDBReviewModel")
