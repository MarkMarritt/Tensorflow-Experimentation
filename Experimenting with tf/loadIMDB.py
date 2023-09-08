from imdbData import *

sample = review[-10:]
label = sentiment[-10:]
dummyText = ["The plot line was very weak and the acting was terrible"]
dummy = pad_sequences(token.texts_to_sequences(dummyText), padding="post", maxlen= 200)

newModel = keras.models.load_model("C:\Code\Bet turtle\ModelSaves\IMDBReviewModel")
result = newModel.predict(dummy)
print(result)
print(label)
#result = newModel.predict( "The Karen Carpenter Story shows a little more about singer Karen Carpenter's complex life. Though it fails in giving accurate facts, and details.<br /><br />Cynthia Gibb (portrays Karen) was not a fine election. She is a good actress , but plays a very naive and sort of dumb Karen Carpenter. I think that the role needed a stronger character. Someone with a stronger personality.<br /><br />Louise Fletcher role as Agnes Carpenter is terrific, she does a great job as Karen's mother.<br /><br />It has great songs, which could have been included in a soundtrack album. Unfortunately they weren't, though this movie was on the top of the ratings in USA and other several countries")
#print(np.mean(result))
#print(dummy)

"""[[0.5449    ]
 [0.66811454]
 [0.566483  ]
 [0.47261474]
 [0.5931908 ]
 [0.6190291 ]
 [0.5191621 ]
 [0.48631305]
 [0.40680084]
 [0.5507003 ]]"""