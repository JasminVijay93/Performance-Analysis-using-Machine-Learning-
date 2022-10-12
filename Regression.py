#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import Regression
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error
from math import sqrt
import statistics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
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
   

class RegressionClass():         
    
    def transform(self,data): 
        
#         #Categorise data into three classes
#         cat = (-1, 10.0, 15.0,20.0)
#         cat_name = ['poor','average','excellent']
#         data['G3']= pd.cut(data['G3'], bins= cat, labels= cat_name)
        
        #Identify categorical features
        classes=data.select_dtypes(include='object')
        class_col=list(classes.columns)        
        le = LabelEncoder() 
        
        #Encoding using label encoder
        for x in data[class_col]:
            data[x]=le.fit_transform(data[x])
        return data
    
    def RandomForestRegressor(self,X_train, X_test, y_train, y_test):   
                        
        #need to increase max_features(for mse) decrease random state for r2        
        #max_features =0.9 ,max_depth=6,min_samples_split=8,n_estimators=200,random_state=0(success)
        
        #max_features =0.9 ,max_depth=6,min_samples_split=5,n_estimators=500,random_state=0
        regr = RandomForestRegressor(max_features =0.5,n_estimators=200)
        # Fitting Random Forest Regression to the dataset 
        model =regr.fit(X_train, y_train)
        
        #Prediction of data using test data
        prediction = model.predict(X_test)
        
        
        #Calculating mean squared error and R2
        mse = mean_squared_error(y_test, prediction)
        rmse = mse**.5
        print(green)
        print(bold)
        print("Mean Squared Error",round(mse,2))
        print("Root Mean Squared Error",round(rmse,2))
        r2=round(r2_score(y_test, prediction),2)
        print("R2 score",r2)
        
        #Cross validation results
        scores = cross_val_score(regr, X_train, y_train, cv=5)
        accuracy1=round(scores.mean()*100,2)        
        print("Accuracy of Random Forest Regressor before gridserachcv",round(scores.mean()*100,2),"%")  
        return model
        
    def scatterplot(self,X_test,y_test,model):
        prediction = model.predict(X_test)
        #The variance between true value and predicted value
        print(bold+"Difference between predicted and expected value"+close)
        d={'expected value' : list(y_test), 'predicted value' : pd.Series(prediction)}
        display(pd.DataFrame(d).head(5)) 
        df=pd.DataFrame(d).head(10)
        y_testdata=df['expected value']
        y_pred=df['predicted value']
                
        #Scattre plot to draw the true and predicted value
        if max(y_testdata) >= max(y_pred):
            my_range = int(max(y_testdata))
        else:
            my_range = int(max(y_pred))
        import matplotlib.pyplot as plt
        plt.scatter(range(len(y_testdata)), y_testdata, color='blue')
        plt.scatter(range(len(y_pred)), y_pred, color='red')
        plt.title("Random Forest Regression")
        plt.show()
        
    def feature_imp(self,X,model):
        #Finding the Feature importance
        feature_importances = pd.DataFrame(model.feature_importances_[:16])
        columnsname=list(X.columns)
        df_feature_importance = pd.DataFrame(model.feature_importances_, index=columnsname, columns=['featureimportance']).sort_values('featureimportance', ascending=False)
        print(df_feature_importance.head(10))  
        df_feature_importance.plot(kind='bar',title='Plots Comparison for Feature Importance')
        plt.show()
        
   
    
    def GridSearchCV(self,X_train, X_test, y_train, y_test):
        #Parameter Grid
        param_grid = { 
            "n_estimators"      : [100,200,500],
            "max_features"      : ["auto", "sqrt", "log2"],
            "min_samples_split" : [2,4,8],
            "max_depth" : [2,4,8],
            "bootstrap": [True, False],
            }
        regr = RandomForestRegressor()    
        grid = GridSearchCV(regr, param_grid, n_jobs= -1, cv=5)
        a=grid.fit(X_train, y_train)
        print("The Best Parameters are",grid.best_params_)
        regressor = RandomForestRegressor().set_params(**grid.best_params_)
        regressor.fit(X_train, y_train)
        prediction = regressor.predict(X_test) 
        
        #Calculating the metrices
        print(red)
        print(bold)
        mse = mean_squared_error(y_test, prediction)        
        rmse = mse**.5 
        print(close)
        print("Mean square error",round(mse,2))       
        print("root mean square error",round(rmse,2))        
        r2=round(r2_score(y_test, prediction),2)
        print("R2 score",r2)
        
        scores = cross_val_score(regressor, X_train, y_train, cv=5)
        accuracy2=round(scores.mean()*100,2)        
        print("Accuracy of Random Forest Regressor after gridserachcv",round(scores.mean()*100,2),"%")    
        print(close)
         
        
