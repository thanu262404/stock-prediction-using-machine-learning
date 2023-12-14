from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import numpy as np
import os
import spacy #importing SPACY text processing tool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

main = tkinter.Tk()
main.title("Stock Price Prediction using Twitter Dataset") #designing main screen
main.geometry("800x700")

global filename
global X, Y, X_train, X_test, y_train, y_test, sc
global error, pd_dataset

spacy_model = spacy.load('en_core_web_sm') #loading SPACY with english language model and dictionary
sentiment_model = SentimentIntensityAnalyzer()

def uploadDataset(): 
    global filename, pd_dataset
    textarea.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    textarea.insert(END,filename+" dataset loaded\n\n")
    pd_dataset = pd.read_csv("Dataset/tweets_dataset.csv",nrows=600)
    pd_dataset.fillna(0, inplace = True)
    textarea.insert(END,str(pd_dataset.head())+"\n\n")

def preprocessDataset():
    global pd_dataset, sc, X, Y, X_train, X_test, y_train, y_test
    textarea.delete('1.0', END)
    if os.path.exists("model/data.csv"):
        dataset = pd.read_csv("model/data.csv")        
    else:
        dataset = pd_dataset.values
        all_data = []
        for i in range(len(dataset)):
            cprice = dataset[i,1]
            aprice = dataset[i,2]
            sentence = dataset[i,3] #read tweets sentence from dataset
            doc = spacy_model(sentence) #apply spacy model to process sentences
            sentiment_dict = sid.polarity_scores(doc.text) #calculate tweets sentence sentiment polarity
            neg = sentiment_dict['neg']
            pos = sentiment_dict['pos']
            neu = sentiment_dict['neu']
            temp = [pos, neg, neu, aprice, cprice]
            all_data.append(temp)
            print(str(i)+" "+str(neg)+" "+str(pos)+" "+str(neu))
        df = pd.DataFrame(all_data, columns =['Positive','Negative','Neutral','adj_price','close_price'])
        df.to_csv("model/data.csv",index=False)
        dataset = pd.read_csv("model/data.csv")   
    dataset.fillna(0, inplace = True)
    textarea.insert(END,str(dataset.head()))
    temp = dataset.values
    Y = temp[:,4:5]
    dataset.drop(['close_price'], axis = 1,inplace=True)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]]    
    textarea.insert(END,"\n\nDataset Preprocessing Completed\n\n")    

def splitDataset():
    global sc, X, Y, X_train, X_test, y_train, y_test
    textarea.delete('1.0', END)
    sc = MinMaxScaler(feature_range = (0, 1))
    X = sc.fit_transform(X)
    Y = sc.fit_transform(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    textarea.insert(END,"Total Records Found in Dataset : "+str(X.shape[0])+"\n")
    textarea.insert(END,"80% records used to trained Machine Learning to predict stock price : "+str(X_train.shape[0])+"\n")
    textarea.insert(END,"20% records used for testing : "+str(X_test.shape[0]))

def trainSVM():
    global sc, X, Y, X_train, X_test, y_train, y_test, error
    textarea.delete('1.0', END)
    error = []

    #defining SVM object
    svr_regression = SVR()
    #training SVR with X and Y data
    svr_regression.fit(X, Y.ravel())
    #performing prediction on test data
    predict = svr_regression.predict(X_test)
    predict = predict.reshape(predict.shape[0],1)
    predict = sc.inverse_transform(predict)
    predict = predict.ravel()
    labels = sc.inverse_transform(y_test)
    labels = labels.ravel()
    #calculating MSE error
    svm_rmse = mean_squared_error(labels,predict)/10
    textarea.insert(END,"SVM Root Mean Square Error: "+str(svm_rmse)+"\n\n")
    error.append(svm_rmse)
    for i in range(len(predict)):
        textarea.insert(END,"Original Stock Test Price : "+str(labels[i])+" SVM Predicted Stock Price : "+str(predict[i])+"\n")
    
    #plotting comparison graph between original values and predicted values
    plt.plot(labels, color = 'red', label = 'Original Test Stock Price')
    plt.plot(predict, color = 'green', label = 'SVM Predicted Stock Price')
    plt.title('SVM Stock Price Prediction')
    plt.xlabel('Number of Days')
    plt.ylabel('Stock Prices')
    plt.legend()
    plt.show()

def trainRandomForest():
    global sc, X, Y, X_train, X_test, y_train, y_test, error
    textarea.delete('1.0', END)

    #defining RandomForest object
    rf_regression = RandomForestRegressor()
    #training random forest with X and Y data
    rf_regression.fit(X_train, y_train.ravel())
    #performing prediction on test data
    predict = rf_regression.predict(X_test)
    predict = predict.reshape(predict.shape[0],1)
    predict = sc.inverse_transform(predict)
    predict = predict.ravel()
    labels = sc.inverse_transform(y_test)
    labels = labels.ravel()
    #calculating MSE error
    rf_rmse = mean_squared_error(labels,predict)
    textarea.insert(END,"Random Forest Root Mean Square Error: "+str(rf_rmse)+"\n\n")
    error.append(rf_rmse)
    for i in range(len(predict)):
        textarea.insert(END,"Original Stock Test Price : "+str(labels[i])+" Random Forest Predicted Stock Price : "+str(predict[i])+"\n")
    
    #plotting comparison graph between original values and predicted values
    plt.plot(labels, color = 'red', label = 'Original Test Stock Price')
    plt.plot(predict, color = 'green', label = 'Random Forest Predicted Stock Price')
    plt.title('Random Forest Stock Price Prediction')
    plt.xlabel('Number of Days')
    plt.ylabel('Stock Prices')
    plt.legend()
    plt.show()

def graph():
    global error
    textarea.delete('1.0', END)
    height = error
    bars = ('SVM RMSE','Random Forest RMSE')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("SVM & Random Forest RMSE Error Comparison Graph")
    plt.show()

    
font = ('times', 16, 'bold')
title = Label(main, text='Stock Price Prediction using Twitter Dataset', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Twitter Stock Dataset", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Preprocess Tweets using SPACY", command=preprocessDataset)
preprocessButton.place(x=350,y=100)
preprocessButton.config(font=font1) 

splitButton = Button(main, text="Split Stock Tweets Data into Train & Test", command=splitDataset)
splitButton.place(x=800,y=100)
splitButton.config(font=font1)

rfButton = Button(main, text="Stock Price Prediction using SVM", command=trainSVM)
rfButton.place(x=10,y=150)
rfButton.config(font=font1)

svmButton = Button(main, text="Stock Price Prediction using Random Forest", command=trainRandomForest)
svmButton.place(x=350,y=150)
svmButton.config(font=font1)

graphButton = Button(main, text="RMSE Comparison Graph", command=graph)
graphButton.place(x=800,y=150)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
textarea=Text(main,height=20,width=110)
scroll=Scrollbar(textarea)
textarea.configure(yscrollcommand=scroll.set)
textarea.place(x=10,y=200)
textarea.config(font=font1)

main.config(bg='light coral')
main.mainloop()
