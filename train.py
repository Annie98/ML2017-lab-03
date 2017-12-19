#coding: utf-8
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from feature import NPDFeature
from ensemble import AdaBoostClassifier

def get_grayscale(path,x=None,y=None):
    for file in os.listdir(path):
        file_path = os.path.join(path,file)
        labels = None
        if path == 'datasets/original/face/':
            labels = [1]
        elif path == 'datasets/original/nonface/':
            labels = [-1]
        image = Image.open(file_path).convert('L').resize((24,24)) #将rgb图片转化成灰度图
        if (x is None) & (y is None):
            x=np.array([NPDFeature(np.asarray(image)).extract()])
            y=np.array([labels])
        else:
            #把脸的和非脸的图片拼接起来
            x=np.vstack((x,NPDFeature(np.asarray(image)).extract()))
            y=np.vstack((y,labels))
    return x,y

def pre_image():
	x,y = get_grayscale('datasets/original/face/')
	x,y = get_grayscale('datasets/original/nonface/',x,y)
	print(x.shape,y.shape)
	with open('features', "wb") as file:
		pickle.dump(x, file)
	with open('labels', "wb") as file:
		pickle.dump(y, file)
  
def acc_plot(validation_score_list):
    plt.title('Adaboost')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.plot(range(len(validation_score_list)),validation_score_list)  
    plt.grid()
    plt.show()
        
if __name__ == "__main__":
    # write your code here    
    #pre_image()
    with open('features', "rb") as f:
        x = pickle.load(f) 
    with open('labels',"rb") as f: 
        y = pickle.load(f)
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)
    maxIteration = 10
    predict_y = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),maxIteration).fit(X_train, y_train).predict(X_test)
    
    with open('report.txt', "wb") as f:
        report = classification_report(y_test,predict_y,target_names=["face","nonface"])
        f.write(report.encode())


