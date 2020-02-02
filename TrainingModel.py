##from numpy import array
##import os
##from pickle import dump
##import pandas as pd
##import pickle
###from keras.preprocessing.text import Tokenizer
###input=r'G:\temporary\token2'
###output=r'G:\temporary\pickledict'
##input=r'final.csv'

###join frames
##inp = pd.read_csv(input)
##print(len(inp))
###print(inp['Label'].nunique())
###print(inp.Label.unique())
###df = inp.groupby('Label').nunique()
###with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
###    print(df)

###tokenizer = Tokenizer()
###inp = pd.read_csv(input)
###inp['Label']=0
####inp['Text']=pd.eval(inp['Text'])

###for i in range(0,inp['Text'].size):
###        p=inp['Text'][i]
###        p=eval(p)
###        lines=' '.join(p)
###        tokenizer.fit_on_texts(lines)

###f=open(os.path.join(output,"thedictionary2.dict"), 'wb')
###pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)


#from numpy import array
#import os
#from pickle import dump
#import pandas as pd
#import pickle
##from keras.preprocessing.text import Tokenizer
##input=r'G:\temporary\token2'
##output=r'G:\temporary\pickledict'
#input=r'C:\Users\Stefan\source\repos\vocabgenerator\vocabgenerator\final.csv'
#input2=r'C:\Users\Stefan\source\repos\vocabgenerator\vocabgenerator\myfile22.csv'
#output=r'C:\Users\Stefan\source\repos\vocabgenerator\vocabgenerator\categorical.csv'

##join frames
#inp = pd.read_csv(input)
#outp=pd.get_dummies(inp['Label'],prefix=['category'])
#outp.to_csv(output)
##print(pd.get_dummies(inp,prefix=['Label']))
##inp2 = pd.read_csv(input2)
##inp['Label']=0
##inp2['Label']=1
##outs=pd.concat([inp,inp2])
##outs.to_csv("finalbinary.csv")


###file = open(input,'rb')
###tokenizer=pickle.load(file)
##tokenizer = Tokenizer()
###inp = pd.read_csv(input)
###inp['Label']=0
####inp['Text']=pd.eval(inp['Text'])

##for i in range(0,inp['Text'].size):
##        p=inp['Text'][i]
##        p=eval(p)
##        lines=' '.join(p)
##        tokenizer.fit_on_texts(lines)

##f=open(os.path.join(output,"thedictionary2.dict"), 'wb')
##pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)


import pandas as pd
import pickle
import os
from keras import optimizers
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import GlobalMaxPooling1D
from keras.layers import Input, Flatten
input=r'C:\Users\Stefan\source\repos\vocabgenerator\vocabgenerator\final.csv'
#from numpy import array
import numpy
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Flatten, Bidirectional
from keras.layers import MaxPooling1D
from keras.layers import SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

print('works')
import ast
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
def boolean(text):
 return text in ['True']
with open('thedictionary.dict', 'rb') as handle:
    tokenizer = pickle.load(handle)
max_function_length = 2021
embedding_vecor_length = 40
inp = pd.read_csv(input)

length=len(inp)
inp.Text = inp.Text.apply(ast.literal_eval)
X_data=inp.sample(n=int(80/100.0*len(inp)), random_state=1)

X_datatest=inp.sample(n=int(20/100.0*length), random_state=1)

cve1 = numpy.array(X_data.Label,dtype=numpy.bool)
X_data=X_data.Text
cve1test=numpy.array(X_datatest.Label,dtype=numpy.bool)
X_datatest=X_datatest.Text
X_datatest=tokenizer.texts_to_sequences(X_datatest)
X_datatest=sequence.pad_sequences(X_datatest, maxlen=max_function_length,padding='post')

#print(len(X_data))
max=0
#for i in range(len(X_data)):
#    if len(X_data[i])>max:
#        max=len(X_data[i])
#print(max)  2021

#cve1=cve1[:int(80/100.0*len(cve1))]
X_data2=tokenizer.texts_to_sequences(X_data)
X_data2=sequence.pad_sequences(X_data2, maxlen=max_function_length,padding='post')
alpha=cve1
#alpha=alpha*1
vocab_size = len(tokenizer.word_index) + 1
model = Sequential()
#model.add(Embedding(vocab_size, embedding_vecor_length, input_length=max_function_length))
#model.add(Bidirectional(LSTM(100)))
#model.add(Dropout(0.5))






#model.add(SpatialDropout1D(0.3))
#model.add(LSTM(256, dropout = 0.3, recurrent_dropout = 0.3))
#model.add(Dense(256, activation = 'relu'))
#model.add(Dropout(0.3))

#model.add(LSTM(100,return_sequences=True))
#model.add(LSTM(100, recurrent_dropout=0.2,return_sequences=True))


model.add(Embedding(vocab_size, embedding_vecor_length, input_length=max_function_length))
model.add(Conv1D(128, 3, strides=1, activation = 'relu'))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])
print(model.summary())
filepath="weights-improvement2.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
model.fit(X_data2, alpha, epochs=2,validation_split=0.30, callbacks=callbacks_list, batch_size=150)
results = model.evaluate(X_datatest, cve1test, batch_size=128)
#model.add(Flatten())
#model.add(Dense(100, activation='relu'))
#model.add(Dense(100,activation='relu'))
#model.add(Dense(100, activation='relu'))
#model.add(Dense(100, activation='relu'))
#model.add(Dense(100,  activation='relu'))
#model.add(Dense(100,  activation='relu'))
#model.add(Dense(100,  activation='relu'))
#model.add(Dense(100,  activation='relu'))
#model.add(Dense(100,  activation='relu'))
#model.add(Dense(100,  activation='relu'))



#model.add(Embedding(vocab_size, embedding_vecor_length, input_length=max_function_length))
#model.add(LSTM(100, recurrent_dropout=0.2,return_sequences=True))
#model.add(LSTM(100, recurrent_dropout=0.2,return_sequences=True))
#model.add(LSTM(100))


print('test loss, test acc:', results)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

