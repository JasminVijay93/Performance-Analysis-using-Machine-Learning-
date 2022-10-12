#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from matplotlib import style



#Color Palette
bold='\033[1m'
close='\033[00m'
red='\033[91m'
newline='\n'
green='\033[92m'
blue='\033[93m'
   
  
                  

class load_data():
               
    def read_csv():
        try:
            
            #Fetching data from excel
            math = pd.read_csv("student-mat.csv",sep=';')
            por= pd.read_csv("student-por.csv",sep=';')
            stud=por.merge(math,how='outer',on=None,indicator=True)
            stud['_merge'] = stud['_merge'].str.replace('left_only','Portuguese').str.replace('right_only','Mathematics')
            stud.rename(columns = {'_merge' : 'Subject'} , inplace = True)
            #stud = pd.concat([por,math])            
            return math,por,stud
        
        except IOError:
            print("File not exist")
        except SyntaxError:
            print("Syntax error encountered")
        
class EDAClass():
    def EDA(self):
        
        #Loading data to dataframe
        data=load_data.read_csv()
        print(bold+"-------------------Exploratory Data Analysis Module-----------------"+close)
        math=data[0]
        por=data[1]
        stud=data[2]
        
        #EDAClass.data_preprocessing(stud)
        
        #EDAClass.descriptive_statistical_analysis(stud) 
        #minidf=EDAClass.mini_dataset(stud)
        
        #EDAClass.visualisation(stud)
        
        #EDAClass.correlation_matrix(stud) 
        
        #stud.drop(['Subject','grade'], axis = 1,inplace = True)
        return stud 

    def mini_dataset(data):
        
        #Selecting important features only
        df_new=data[['G3','G2','G1','absences','studytime','failures','famrel','age','goout','health','guardian']]
        return df_new
    
    def data_preprocessing(df):
        
        #Checking Null values
        a=df.isnull().any()
        b=bool(a.unique())
        if (b == False):
            print(bold+green+"\nThere is no missing values in the dataset!!!!!!!!!!!!!!!!!! \n " +close)
        else:
            print(bold+red+"\n Null values present ,cleansing required ! \n "+close) 
        
        print(bold+blue+"Insights of data\n"+close)
        
        #Providing high level information of data
        print("The number of students in each school:\n",df['school'].value_counts())
        print("************************************************************************")
        print("The mean value of final grade for male and female:\n",df.groupby('sex')['G3'].mean())
        print("************************************************************************")
        print("The number of students in each subjects \n",df['Subject'].value_counts())
        print("************************************************************************")
        print("The mean value of final grade for male and female:\n",df.groupby('Subject')['G3'].mean())
        print("************************************************************************")        
        print("Number of students who are not attending the exam:\n",len(df.loc[df['G3'] == 0]))
        print("************************************************************************")
             
 

    def categorise(stud):        
        stud_category =stud
        #Let's categorise all the contionous variables to categorical variable 
        cat = (-1, 10.0, 15.0,20.0)
        cat_name = ['poor','average','excellent']
        stud['grade']= pd.cut(stud['G3'], bins= cat, labels= cat_name)        
        return stud_category             
    

            
    def descriptive_statistical_analysis(data):
        
        #Identify all numeric columns
        result = data.select_dtypes(include='number')
        col_list=result.columns
        list=col_list.to_numpy()
        d=[]
        dlist=[]
        df=pd.DataFrame()       
        for x in list:  
            d=[round(data[x].mean(),2),round(data[x].median(),2),round(data[x].std(),2),round(data[x].var(),2),
               round(data[x].min(),2),round(data[x].max(),2),round(data[x].skew(),2),round(data[x].kurtosis(),2),
              round(data[x].count(),2),round(data[x].quantile(q=0.25),2),round(data[x].quantile(q=0.5),2),
               round(data[x].quantile(q=0.75),2)]
            dlist.append(d)
        df=pd.DataFrame(dlist, columns = ['Mean', 'Median','Std' ,'Variance','Minimum','Maximum','Skewness','Kurtosis','Count','25%','50%','75%'], index=col_list )
        print(bold+"Descriptive Statistical Analysis")
        print("************************************************************************")
        display(df)
        
        
        
    def correlation_matrix(data):        
        # find the correlations
        cormatrix = data.corr()
        cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T
        # find the top n correlations
        cormatrix = cormatrix.stack()
        cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index()
        print("************************************************************************")
        cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]
        print(cormatrix.head(10))
        #correlation between variables through a correlation heatmap
        print("************************************************************************")
        print(bold+blue+"Correlation Matrix of G3 \n"+close)
        corr_matrix = data.corr()
        print(corr_matrix["G3"].sort_values(ascending=False))
        plt.figure(figsize=(10,10))
        sns.heatmap(corr_matrix, annot=True, cmap="Blues",fmt=".2f")
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns);
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.title('Correlation HeatMap', fontsize=10)
        plt.show()      

    
    def gradeandfeature(data):
        
        perc = (lambda col: col/col.sum())
        index = ["poor","average","excellent"]
        relationship_index = pd.crosstab(index=data['grade'], columns=data['internet'])
        romantic_index = relationship_index.apply(perc).reindex(index)
        romantic_index.plot.bar(colormap='mako_r',fontsize=10, figsize=(5,4))
        plt.title('Grade By Internet Usage', fontsize=10)
        plt.ylabel('Percentage of Students', fontsize=10)
        plt.xlabel('Final Grade', fontsize=10)
        plt.show()
        print(blue +"Internet usage helps to score excellent grade"+close)
        print("************************************************************************")
        
        perc = (lambda col: col/col.sum())
        index = ["poor","average","excellent"]
        relationship_index = pd.crosstab(index=data['grade'], columns=data['address'])
        romantic_index = relationship_index.apply(perc).reindex(index)
        romantic_index.plot.bar(colormap='mako_r',fontsize=10, figsize=(5,4))
        plt.title('Grade By Demography', fontsize=10)
        plt.ylabel('Percentage of Students', fontsize=10)
        plt.xlabel('Final Grade', fontsize=10)
        plt.show()
        print("Students who is living in urban area score excellent grade")
        print("************************************************************************")
        
        
        perc = (lambda col: col/col.sum())
        index = ["poor","average","excellent"]
        relationship_index = pd.crosstab(index=data['grade'], columns=data['Dalc'])
        romantic_index = relationship_index.apply(perc).reindex(index)
        romantic_index.plot.bar(colormap='mako_r',fontsize=8, figsize=(5,4))
        plt.title('Grade By Alchol Consumption', fontsize=8)
        plt.ylabel('Percentage of Students', fontsize=8)
        plt.xlabel('Final Grade', fontsize=10)
        plt.show()
        print("The alchol consumption have dependency with acdemic growth")
        print("************************************************************************")
        
        
        perc = (lambda col: col/col.sum())
        index = ["poor","average","excellent"]
        relationship_index = pd.crosstab(index=data['grade'], columns=data['studytime'])
        romantic_index = relationship_index.apply(perc).reindex(index)
        romantic_index.plot.bar(colormap='mako_r',fontsize=10, figsize=(5,4))
        plt.title('Grade By Studytime', fontsize=8)
        plt.ylabel('Percentage of Students', fontsize=10)
        plt.xlabel('Final Grade', fontsize=10)
        plt.show()
        print("The excellant students are studying for hours")
        print("************************************************************************")
        
        plt.figure(figsize=(4,5))
        sns.boxplot(x='goout', y='G3', data=data, palette='hot')
        plt.title('Final Grade By Frequency of Going Out', fontsize=10)
        plt.ylabel('Final Score', fontsize=16)
        plt.xlabel('Frequency of Going Out', fontsize=16)
        print("************************************************************************")

        
        perc = (lambda col: col/col.sum())
        index = ["poor","average","excellent"]
        relationship_index = pd.crosstab(index=data['grade'], columns=data['famrel'])
        romantic_index = relationship_index.apply(perc).reindex(index)
        romantic_index.plot.bar(colormap='mako_r',fontsize=8, figsize=(8,5))
        plt.title('Grade By Parental Status', fontsize=8)
        plt.ylabel('Percentage of Students', fontsize=8)
        plt.xlabel('Final Grade', fontsize=8)
        plt.show()
        print("Parental status have influence in childhren's studies")     
        print("**************************************************************************************************")
        
            
                
        perc = (lambda col: col/col.sum())
        index = ["GP","MS"]
        relationship_index = pd.crosstab(index=data['school'], columns=data['Pstatus'])
        romantic_index = relationship_index.apply(perc).reindex(index)
        romantic_index.plot.bar(colormap='mako_r',fontsize=10, figsize=(8,5))
        plt.title('School By Alchol consumption', fontsize=10)
        plt.ylabel('Percentage of Consumption', fontsize=10)
        plt.xlabel('School', fontsize=10)
        plt.show()
        print("************************************************************************")
        
        perc = (lambda col: col/col.sum())
        index = ["GP","MS"]
        relationship_index = pd.crosstab(index=data['school'], columns=data['G3'])
        romantic_index = relationship_index.apply(perc).reindex(index)
        romantic_index.plot.bar(colormap='mako_r',fontsize=10, figsize=(8,5))
        plt.title('School By G3', fontsize=8)
        plt.ylabel('Percentage of Marks', fontsize=8)
        plt.xlabel('School', fontsize=8)
        plt.show()
        
        print("************************************************************************")
        
        perc = (lambda col: col/col.sum())
        index = ["GP","MS"]
        relationship_index = pd.crosstab(index=data['school'], columns=data['romantic'])
        romantic_index = relationship_index.apply(perc).reindex(index)
        romantic_index.plot.bar(colormap='mako_r',fontsize=8, figsize=(8,5))
        plt.title('School By Romance', fontsize=8)
        plt.ylabel('Percentage', fontsize=8)
        plt.xlabel('School', fontsize=8)
        plt.show()
        print("************************************************************************")
        
        
        
    
    def piechart(data):
        style.use('ggplot') 
        print(bold+blue+"Mjob Distribution"+close)
        labels = data['Mjob'].unique()
        
        x=[len(data.loc[(data.Mjob=='at_home')]),
           len(data.loc[(data.Mjob=='health')]),
           len(data.loc[(data.Mjob=='other')]),
           len(data.loc[(data.Mjob=='services')]),
           len(data.loc[(data.Mjob=='teacher')]),
           ]
        color = ['red','yellow','blue','coral','lightskyblue']
        plt.pie(x, labels=labels,colors=color,autopct ="%1.1f%%")
        plt.show()
              
    def visualisation(data):
        data=EDAClass.categorise(data)       
        
        print(bold+"Frequency plots"+close)
        print("************************************************************************")
        
        f, ax = plt.subplots()
        figure = sns.countplot(x = 'school', data=data, order=['GP','MS'], palette='Set3')
        ax = ax.set(ylabel="Count", xlabel="school")
        figure.grid(False)
        plt.title('School Distribution')
        plt.show()
        print(bold+"GP is having more students than MS")
        print("************************************************************************")
        
        f, ax = plt.subplots()
        figure = sns.countplot(x = 'Subject', data=data, order=['Portuguese','Mathematics'], palette='Set3')
        ax = ax.set(ylabel="Count", xlabel="Subject")
        figure.grid(False)
        plt.title('Subject Distribution')
        plt.show()
        print(bold+"Portuguese data is having more students than Mathematics")
        print("************************************************************************")
        
        f, ax = plt.subplots()
        figure = sns.countplot(x = 'sex', data=data, order=['M','F'], palette='Set3')
        ax = ax.set(ylabel="Count", xlabel="gender")
        figure.grid(False)
        plt.title('Gender Distribution')
        plt.show()
        print(bold+"Female students are more than Male students")
        print("************************************************************************")
        
        f, ax = plt.subplots()
        figure = sns.countplot(x = 'address', data=data, order=['U','R'], palette='Set3')
        ax = ax.set(ylabel="Count", xlabel="address")
        figure.grid(False)
        plt.title('Address Distribution')
        plt.show()
        print(bold+"Urban students are more than Rural students")
        print("************************************************************************")
    
        f, ax = plt.subplots()
        figure = sns.countplot(x =data['G3'], data=data, palette='Set3')
        ax = ax.set(ylabel="Count", xlabel="G3")
        figure.grid(False)
        plt.title('Final grade Distribution')
        plt.show()  
        print(bold+'G3 is normally distributed apart from absent students')
        print("************************************************************************")
        
        f, ax = plt.subplots()
        figure = sns.countplot(x =data['reason'], data=data, palette='Set3')
        ax = ax.set(ylabel="Count", xlabel="G3")
        figure.grid(False)
        plt.title('Reason for choosing the school')
        plt.show()  
        print(bold+'The main reason for choosing the school')
        print("************************************************************************")
        
             
        
        f, ax = plt.subplots()
        figure=sns.countplot(data['grade'], order=["poor","average","excellent"], palette='Set1')
        ax = ax.set(ylabel="Count", xlabel="grade")
        figure.grid(False)
        plt.title('Grade Classification')
        plt.show()  
        print(bold+"Average students er more than excellent and low students")
        print("************************************************************************")
        
        print("\n Dependency of G3 with other columns\n")
        print("************************************************************************")
        
        #function to plot G3 and other columns
        EDAClass.gradeandfeature(data)   
        EDAClass.piechart(data)         
        print("EDA completed........ !!!!")
        
        
       


