#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from  sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import Regression
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error
from math import sqrt
import statistics
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score




#Color Palette
bold='\033[1m'
close='\033[00m'
red='\033[91m'
newline='\n'
green='\033[92m'
blue='\033[93m'
an='\033[94m'


class Classify():
    
            
    
    def RandomForstClassifier(self,X_train, X_test, y_train, y_test):
            # code for random forestclassifier
             
             #copy of df to avoid overwrite
           
            #n_estimators=1000,max_depth=2, random_state=0  ,max_features =0.9 ,max_depth=6,min_samples_split=8, 
            
            rd =  RandomForestClassifier(n_estimators=200,random_state=110)
            rd_model = rd.fit(X_train,y_train)
            predict_rd = rd_model.predict(X_test) 
            
            #Calculating metrices
            print(green)
            print(bold)
            scores = cross_val_score(rd_model, X_train, y_train, cv=5)
            accuracy1=round(scores.mean()*100,2)
            print("Accuracy of Random Forest Model without sampling",round(scores.mean()*100,2),"%")
            print("************************************************************************")
            
            print("Classification Report of Random Forest Classification (without sampling) \n")
            print(classification_report(y_test,predict_rd))        
            print("Confusion Matrix of Random Forest Classification (without sampling)\n",confusion_matrix(y_test,predict_rd))
            fig = plot_confusion_matrix(rd, X_test, y_test, display_labels=rd.classes_)
            fig.figure_.suptitle("Confusion Matrix of Random Forest Classification\n ")
            plt.show()
            print("************************************************************************")
            


          


            print("Difference between predicted and expected value")
            d={'expected value' : list(y_test), 'predicted value' : pd.Series(predict_rd)}
            display(pd.DataFrame(d).head(10))
            print("************************************************************************")
            


          
        # Sampling the data using SMOTE
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            #np.bincount(y_res)        
            rd =  RandomForestClassifier(bootstrap= True, max_features='auto',min_samples_split= 8, n_estimators= 20)
            #Fitting the training data
            rd_model = rd.fit(X_res,y_res)
            #Predicting on test
            predict_rd = rd_model.predict(X_test) 
            
            #Calculating the metrics
            scores = cross_val_score(rd_model, X_train, y_train, cv=5)
            accuracy2=round(scores.mean()*100,2)
            print("Accuracy of Random Forest Classifier Model after sampling(SMOTE)",round(scores.mean()*100,2),"%")
            print("Classification Report of Random Forest Classification (SMOTE) \n")
            print(classification_report(y_test,predict_rd))        
            print("Confusion Matarix of Random Forest Classification",confusion_matrix(y_test,predict_rd))
            fig = plot_confusion_matrix(rd, X_test, y_test, display_labels=rd.classes_)
            fig.figure_.suptitle("Confusion Matrix of Random Forest Classification - SMOTE\n ")
            plt.show()           
            
            print("************************************************************************")
            
           
            #Sampling data using Random Under Sampling
            rus = RandomUnderSampler(random_state=42)
            X_res, y_res = rus.fit_resample(X_train, y_train)        
            rd =  RandomForestClassifier(bootstrap= True, max_features='auto',min_samples_split= 8, n_estimators= 20)     
            #Fitting the training data        
            rd_model = rd.fit(X_res,y_res)
            #Predicting on test        
            predict_rd = rd_model.predict(X_test) 
            scores = cross_val_score(rd_model, X_train, y_train, cv=5)
            accuracy3=round(scores.mean()*100,2)
            print("Accuracy of Random Forest Classifier Model after Random Under Sampling",round(scores.mean()*100,2),"%")
            print(classification_report(y_test, predict_rd))            
            print(close)
            print("************************************************************************")
            return accuracy3
            

    def SVM(self,X_train, X_test, y_train, y_test): 
        
        #SVM Classifier
        svc = SVC(kernel='rbf', C=1, gamma='auto')
        svc_model = svc.fit(X_train,y_train)
        predict_svc = svc_model.predict(X_test)
        
        
        #Calculating the metrics
        print(red)
        print(bold)
        print("Classification Report of SVM\n")
        scores = cross_val_score(svc_model, X_train, y_train, cv=5)
        accuracy1=round(scores.mean()*100,2)
        print("Accuracy of SVM model:",round(scores.mean()*100,2),"%")
        print(classification_report(y_test,predict_svc))
        #The confusion Matrix
        fig = plot_confusion_matrix(svc, X_test, y_test, display_labels=svc.classes_)
        fig.figure_.suptitle("Confusion Matrix of SVM ")
        plt.show()
        print("Confusion Matrix\n",confusion_matrix(y_test,predict_svc))        
        print("************************************************************************")    
    
    
        # Sampling using SMOTE
        sm = SMOTE(random_state=100)
        X_smote, y_smote = sm.fit_resample(X_train, y_train)
        #np.bincount(y_res)        
        s = SVC(kernel='rbf', C=1, gamma='auto')
        #Fitting the training data
        model = s.fit(X_smote, y_smote)
        #Predicting on test
        predict_s = model.predict(X_test)
        scores = cross_val_score(model, X_smote, y_smote, cv=5)
        accuracy2=round(scores.mean()*100,2)
        print("Accuracy of SVM Model after sampling(SMOTE)",round(scores.mean()*100,2),"%")
        print(classification_report(y_test,predict_s))        
        print("************************************************************************")
            
        
        #Sampling using Random Sampling
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X_train, y_train)        
        svc = SVC(kernel='rbf', C=1, gamma='auto')    
        #Fitting the training data        
        svc_model = svc.fit(X_res,y_res)
        #Predicting on test        
        predict_svc = svc_model.predict(X_test)
        scores = cross_val_score(svc_model, X_res, y_res, cv=5)
        accuracy3=round(scores.mean()*100,2)
        print("Accuracy of SVM Model after RandomUnder Sampling",round(scores.mean()*100,2),"%")
        print(confusion_matrix(y_test, predict_svc))        
        print(classification_report(y_test, predict_svc))        
        print(close)
        return accuracy3
       


    def MLP(self,X_train, X_test, y_train, y_test):
        mlp_clf = MLPClassifier(hidden_layer_sizes=(150,100,50),max_iter = 300,activation = 'relu',solver = 'adam')
        mlp_model=mlp_clf.fit(X_train, y_train)
        predict_mlp = mlp_clf.predict(X_test)
        
        
        #Calculating metrics
        print(green)
        print(bold)
        accurancy=accuracy_score(y_test,predict_mlp)*100  
        accuracy1=accuracy_score(y_test,predict_mlp)*100
        print('Accuracy of MLP : {:.2f}'.format(round(accuracy_score(y_test,predict_mlp)*100),2))
        fig = plot_confusion_matrix(mlp_clf, X_test, y_test, display_labels=mlp_clf.classes_)
        fig.figure_.suptitle("Confusion Matrix of MLP  (no sampling and gridserachCV)")
        plt.show()
        print(bold+"Classification Report of MLP (no sampling and gridserachCV)\n")
        print(classification_report(y_test, predict_mlp))
        
        
        
        #Sampling using SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        #np.bincount(y_res)        
        mlp_smote = MLPClassifier(hidden_layer_sizes=(150,100,50),max_iter = 300,activation = 'relu',solver = 'adam')
        #Fitting the training data
        mlp_smote_model = mlp_smote.fit(X_res,y_res)
       #Predicting on test
        predict_rd = mlp_smote_model.predict(X_test)
        
        #Calculating metrics
        scores = cross_val_score(mlp_smote_model, X_train, y_train, cv=5)
        accuracy2=round(scores.mean()*100,2)
        print("Accuracy of MLP Model after sampling(SMOTE)",round(scores.mean()*100,2),"%")
        print("Classification Report of MLP - SMOTE \n")
        print(classification_report(y_test, predict_rd))
        fig = plot_confusion_matrix(mlp_smote, X_test, y_test, display_labels=mlp_clf.classes_)
        fig.figure_.suptitle("Confusion Matrix of MLP - SMOTE")
        plt.show()
        
        print("************************************************************************")
            
        
        #Sampling using Random Under Sampling
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X_train, y_train)        
        m = MLPClassifier(hidden_layer_sizes=(150,100,50),max_iter = 50)     
        #Fitting the training data        
        m_model = m.fit(X_res,y_res)
        #Predicting on test        
        predict_rd = m_model.predict(X_test) 
        scores = cross_val_score(m_model, X_train, y_train, cv=5)
        accuracy3=round(scores.mean()*100,2)
        print("Accuracy of MLP Model after Random Under Sampling",round(scores.mean()*100,2),"%")
        print("Classification Report of MLP - random under sampling \n")
        print(confusion_matrix(y_test, predict_rd))        
        print(classification_report(y_test, predict_rd))        
        
        
        param_grid = {
             'hidden_layer_sizes': [(150,100,50), (120,80,40), (200,50,30)],
             'max_iter': [50, 100, 150],
             'activation': ['tanh', 'relu'],
             'solver': ['sgd', 'adam'],
             'alpha': [0.0001, 0.05],
             'learning_rate': ['constant','adaptive'],
            }
        grid = GridSearchCV(mlp_smote, param_grid, n_jobs= -1, cv=5)
        grid.fit(X_train, y_train)
        print(grid.best_params_) 
        grid_predictions = grid.predict(X_test) 
        print(classification_report(y_test, grid_predictions))
        accuracy=accuracy_score(y_test, grid_predictions)*100
        print(bold +' Accuracy of MLP after tunning in SMOTE dataset: {:.2f}'.format(accuracy_score(y_test, grid_predictions)*100))  
        print(close)    
        return accuracy3
     
       

    #def reccomdation(self,accuracy_regression,accuracy_classification,accuracy_SVM,accuracy_MLP):
# print(bold+"Accurancy of Random Forest Regression : " ,accuracy_regression,"%")
# print(bold+"Accurancy of Random Forest Classification : " ,accuracy_classification,"%")
# print(bold+"Accurancy of SVM : " ,accuracy_SVM,"%")
# print(bold+"Accurancy of MLP : " ,accuracy_MLP,"%")





