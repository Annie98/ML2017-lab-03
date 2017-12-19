import pickle
import numpy as np

class AdaBoostClassifier:

    def __init__(self, weak_classifier, n_weakers_limit):
        self.weak_clf = weak_classifier #弱分类器的类
        self.limit = n_weakers_limit #弱分类器的最大数量
        self.G = [] #弱分类器的集合
        self.num = -1 #弱分类器的数量
        self.alpha = []
        self.validation_score_list = []

    def fit(self,X,y): 
        self.W = np.ones(X.shape[0] )/X.shape[0] #初始化样本权值分布
        self.Score = np.zeros(X.shape[0])
        for i in range(self.limit):
            clf = self.weak_clf.fit(X,y,sample_weight = self.W)
            self.G.append(clf)
            P = clf.predict(X)  #采用弱分类器得到的分类结果
            error = np.sum((P != y.reshape(-1,)) * self.W)#分类误差率 
            #error = 1.0 - self.G[i].score(self.X,self.y) 
            self.validation_score_list.append(1.0-error)
            if error > 0.5:
                continue
            elif error == 0:
                break
            e = 0.5*np.log((1.0-error)/error)
            self.alpha.append(e)
            Z = np.multiply(self.W,np.exp(-self.alpha[i]* np.multiply(y.reshape(-1,) , P))) #规范化因子
            self.W = Z/np.sum(Z)
        return self,self.validation_score_list
   
    def predict_scores(self, X):
        Score = np.zeros(X.shape[0])
        if self.num == -1:
            self.num = self.limit
        for i in range(self.num):
            Score += self.alpha[i] * self.G[i].predict(X).flatten(1)
        return Score

    def predict(self, X, threshold=0):
        predict_y = np.zeros(X.shape[0])
        Score = self.predict_scores(X)
        predict_y[np.where(Score > threshold)] = 1
        predict_y[np.where(Score <= threshold)] = -1
        return predict_y
  

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
