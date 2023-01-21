from Models import Sudokunet
import tensorflow as tf
import sklearn as skl
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",required=True,help="path to output model after training")
args=vars(ap.parse_args())

INIT_LR= 1e-3
EPOCHS=10
BS=128

print("[INFO] Accessing MNIST...")
((trainData, trainLabels), (testData, testLabels)) = tf.keras.datasets.mnist.load_data()
trainData=trainData.reshape((trainData.shape[0],28,28,1))
testData=testData.reshape((testData.shape[0],28,28,1))

trainData=trainData.astype("float32")/255.0
testData=testData.astype("float32")/255.0


lb=LabelBinarizer()
trainlabels=tf.one_hot(trainLabels.astype(np.int32), depth=10)
testlabels=tf.one_hot(testLabels.astype(np.int32), depth=10)

print("[INFO] compiling model...")
optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR)
model=Sudokunet.SudokuNet.build(width=28,height=28, depth=1, classes=10)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

print("[INFO] training network...")
H=model.fit(trainData,trainLabels,validation_data=(testData,testLabels),batch_size=BS,epochs=EPOCHS,verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testData)
print(predictions)
'''
ytest=np.argmax(testLabels, axis=0)
ypred=np.argmax(predictions, axis=1)
print(classification_report(
	ytest,ypred))'''

print("[INFO] serializing digit model...")
model.save(args["model"], save_format="h5")

