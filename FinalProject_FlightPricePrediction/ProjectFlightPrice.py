#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Project:Flight Price Prediction Web Application
#Program By:Ayush Pandey
#Email Id:1805290@kiit.ac.in
#DATE:29-Oct-2021
#Python Version:3.7
#CAVEATS:None
#LICENSE:None


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#CSV TO JSON for the database
import csv 
import json
import time

def csv_to_json(csvFilePath, jsonFilePath):
    jsonArray = []
      
    #read csv file
    with open(csvFilePath, encoding='utf-8') as csvf: 
        #load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf) 

        #convert each csv row into python dict
        for row in csvReader: 
            #add this python dict to json array
            jsonArray.append(row)
  
    #convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf: 
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)
          
csvFilePath = r'Flight_Prediction.csv'
jsonFilePath = r'Flight_JSON.json'

csv_to_json(csvFilePath, jsonFilePath)


# In[4]:


import pymongo
client=pymongo.MongoClient()
#Database name
db=client['FlightPrediction']
#Collection name
Flights=db['Flights']


# In[5]:


with open('Flight_JSON.json') as f:
    data = json.load(f)
Flights.insert_many(data)


# In[ ]:





# In[6]:


#Reading the csv file
df=pd.read_csv("Flight_Prediction.csv")


# In[7]:


df.head()


# In[8]:


df.shape


# In[9]:


#Checking for nan values
df.isna().sum()


# In[10]:


df[(df['Route'].isna() | df['Total_Stops'].isna())]
#This shows the nan value are present only in one row


# In[11]:


#Only one nan row so dropping the row and keeping in df1
df1=df[~(df['Route'].isna())]


# In[12]:


df.shape


# In[13]:


df1.isna().sum()


# In[14]:


df1['Source'].unique()


# In[15]:


df1['Destination'].unique()


# In[16]:


#Extracting the data from date and time


# In[17]:


import datetime as dt
df1['Journey_Date']=pd.to_datetime(df1.Date_of_Journey,format="%d/%m/%Y").dt.day
df1['Journey_Month']=pd.to_datetime(df1.Date_of_Journey,format="%d/%m/%Y").dt.month


# In[18]:


df1.drop('Date_of_Journey',inplace=True,axis=1)


# In[19]:


df1.head(2)


# In[20]:


#Extarcting the hour and minute from the departure column
df1['Dep_Hour']=pd.to_datetime(df1.Dep_Time).dt.hour
df1['Dep_Min']=pd.to_datetime(df1.Dep_Time).dt.minute


# In[21]:


#Now droping the dep_time column
df1.drop('Dep_Time',axis=1,inplace=True)


# In[22]:


df1.head()


# In[23]:


#Similarly for arrival time
df1['Arrival_Hour']=pd.to_datetime(df1.Arrival_Time).dt.hour
df1['Arrival_Min']=pd.to_datetime(df1.Arrival_Time).dt.minute
df1.drop('Arrival_Time',inplace=True,axis=1)


# In[24]:


df1.head(2)


# In[25]:


df1.dtypes


# In[26]:


#Now to convert duration to int type we have to iterate to each element present


# In[27]:


def get(n):
    s=n.split(" ")
    if(len(s)==2):
        w=s[0]
        e=s[1]
        w=w[0:len(w)-1]
        e=e[0:len(e)-1]
        w="".join(w)
        w=int(w)
        e="".join(e)
        e=int(e)
        Duration_Hour.append(w)
        Duration_Min.append(e)
    else:
        q=list(n)
        if('h' in q):
            ans=q[0:len(q)-1]
            ans="".join(ans)
            ans=int(ans)
            Duration_Hour.append(ans)
            Duration_Min.append(0)
        else:
            ans=q[0:len(q)-1]
            ans="".join(ans)
            ans=int(ans)
            Duration_Hour.append(0)
            Duration_Min.append(ans)

Duration_Hour=[]
Duration_Min=[]
z=df1['Duration'].apply(get)
df1['Duration_Hour']=Duration_Hour
df1['Duration_Min']=Duration_Min


# In[28]:


df1.head()


# In[29]:


df1.drop('Duration',axis=1,inplace=True)


# In[30]:


df1.head(1)


# In[31]:


#Extracting inferences from categorical data


# In[32]:


df1['Airline'].value_counts()
#All are nominal data so doing One-Hot Encoding


# In[33]:


#Count of Airline
sns.countplot(x='Airline',data=df1,order=df1['Airline'].value_counts().index)
plt.xticks(rotation=90)
plt.title("Total Count Of Airline")
plt.savefig('AirlineCount.png',format='png',bbox_inches='tight')
plt.show()


# In[34]:


#To campare each and every Airline and its price
sns.catplot(y="Price",x="Airline",data=df1.sort_values("Price",ascending=False),kind="box",height=6,aspect=3)
plt.xticks(rotation=90)
plt.title("Airlines and Their Price Distribution")
plt.savefig('Airline_Price.png',format='png',bbox_inches='tight')
plt.show()
#Jet Airway Business has the highest price while all other are kind of same


# In[35]:


#Sorce is also a categorical feature and is nominal
df1['Source'].value_counts()


# In[36]:


#Count of Airline
sns.countplot(x='Source',data=df1,order=df1['Source'].value_counts().index)
plt.xticks(rotation=90)
plt.title("Total Count Of Source")
plt.savefig('SourceCount.png',format='png',bbox_inches='tight')
plt.show()


# In[37]:


#Again using cat plot for Source column
sns.catplot(x="Source",y="Price",data=df1.sort_values("Price",ascending=False),kind="box",height=6,aspect=3)
plt.title("Source And Their Distribution")
plt.savefig('Source_Price.png',format='png',bbox_inches='tight')
plt.show()


# In[38]:


df1.head()


# In[39]:


#Same for Destination
df1['Destination'].value_counts()


# In[40]:


#Count of Airline
sns.countplot(x='Destination',data=df1,order=df1['Destination'].value_counts().index)
plt.xticks(rotation=90)
plt.title("Total Count Of Destination")
plt.savefig('DestinationCount.png',format='png',bbox_inches='tight')
plt.show()


# In[41]:


#Again using cat plot for Source column
sns.catplot(x="Destination",y="Price",data=df1.sort_values("Price",ascending=False),kind="box",height=6,aspect=3)
plt.title("Destination And Their Price Distribution")
plt.savefig('Destination_Price.png',format='png',bbox_inches='tight')


# In[42]:


df1['Additional_Info'].value_counts()


# In[43]:


#As No info contains more than 80% of the data so droping it
df1.drop(['Route','Additional_Info'],axis=1,inplace=True)


# In[44]:


df1.shape


# In[45]:


df1.head(2)


# In[46]:


df1.Total_Stops.unique()


# In[47]:


#Now handling the Total_Stops columns and only getting the numerical data
def handle(n):
    s=n.split(" ")
    #print(s)
    if(len(s)<=1):
        Stops.append(int(0))
    else:
        Stops.append(int(s[0]))
Stops=[]
z=df1['Total_Stops'].apply(handle)
df1['Total_Stops']=Stops


# In[48]:


df1.Total_Stops.unique()


# In[49]:


df1.head()


# In[50]:


#Total stops countplot
sns.countplot(df1['Total_Stops'])
plt.title("Total Stops And Their Count")
plt.savefig('StopCount.png',format='png',bbox_inches='tight')
plt.show()
#Maximum number of stops is 1
#Only one flight with 4 stop


# In[51]:


#Again using cat plot for Source column
sns.catplot(x="Total_Stops",y="Price",data=df1.sort_values("Price",ascending=False),kind="box",height=6,aspect=3)
plt.title("Stops Along with Their Price Distribution")
plt.savefig('Stop_Price.png',format='png',bbox_inches='tight')
plt.show()


# In[52]:


df1['Total_Stops'].value_counts()


# In[53]:


df1.head()


# In[54]:


#Flights maximum on which date
sns.countplot(df1['Journey_Date'])
plt.title("Count of Flight VS Date")
plt.savefig('JourneyCount.png',format='png',bbox_inches='tight')
plt.show()


# In[55]:


#Price on flight on each day
sns.barplot(df1['Journey_Date'], df1['Price'])
plt.title('Days vs Price', size=30)
plt.xticks(rotation=90)
plt.savefig('Date_Price.png',format='png',bbox_inches='tight')
plt.show()


# In[56]:


df1['Journey_Date'].value_counts()


# In[57]:


#Count of flight each month
#Flights maximum on which date
sns.countplot(df1['Journey_Month'])
plt.title("Count of Flight VS Date")
plt.show()


# In[58]:


#As all the month are in numbrs so extracting the month to plot a graph between count of flight each month
df2=pd.DataFrame()
df2['Month']=df1['Journey_Month']
df2['Month'] = df2['Month'].map({1:'JAN',2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',7:'JUL',8:'AUG',9:'SEP',10:'OCT',11:'NOV',12:'DEC'})
df2['Price']=df1['Price']


# In[59]:


#Count of flight each month
#Flights maximum on which date
sns.countplot(df2['Month'])
plt.title("Count of Flight VS Month")
plt.savefig('MonthCount.png',format='png',bbox_inches='tight')
plt.show()


# In[60]:


sns.relplot(x='Month',y='Price',data=df2.sort_values('Price',ascending=True),kind="line")
plt.title("Price Vs Month")
plt.savefig('PriceMonth.png',format='png',bbox_inches='tight')
plt.show()


# In[61]:


df2['Month'].value_counts()


# In[62]:


#For price column checking the skewness
sns.distplot(df1['Price'])
plt.title("Right Skewed")
plt.savefig('PriceSkewed.png',format='png',bbox_inches='tight')
plt.show()
print(df1['Price'].min())
print(df1['Price'].max())
print(df1['Price'].mean())


# In[63]:


plt.boxplot(df1['Price'])
plt.title("Box-Plot Of Price Column")
plt.savefig('PriceBoxPlot.png',format='png',bbox_inches='tight')
plt.show()


# In[64]:


df1.dtypes


# In[65]:


x = df1.drop('Price', axis=1)
y = df1['Price']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
x.head()


# In[66]:


#For categorical columns using ONEHOTENCODING and pipeline so that the size of columns does not increase
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


# In[67]:


ole=OneHotEncoder()
ole.fit(x[['Airline','Source','Destination']])


# In[68]:


values=make_column_transformer((OneHotEncoder(ole.categories_),['Airline','Source','Destination']),remainder='passthrough')


# In[69]:


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, HuberRegressor, LogisticRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import metrics


# In[70]:


models = [['LinearRegression : ', LinearRegression()],
          ['ElasticNet :', ElasticNet()],
          ['Lasso : ', Lasso()],
          ['Ridge : ', Ridge()],
          ['KNeighborsRegressor : ', KNeighborsRegressor()],
          ['DecisionTreeRegressor : ', DecisionTreeRegressor()],
          ['RandomForestRegressor : ', RandomForestRegressor()],
          ['SVR : ', SVR()],
          ['AdaBoostRegressor : ', AdaBoostRegressor()],
          ['GradientBoostingRegressor : ', GradientBoostingRegressor()],
          ['ExtraTreeRegressor : ', ExtraTreeRegressor()],
          ['HuberRegressor : ', HuberRegressor()],
          ['BayesianRidge : ', BayesianRidge()]]


# In[71]:


for name, model in models:
    model=model
    pipe=make_pipeline(values,model)
    pipe.fit(x_train, y_train)
    predictions = pipe.predict(x_test)
    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))


# In[72]:


#Doing Linear regression for predicting
from sklearn.linear_model import LinearRegression
model=LinearRegression()
pipe=make_pipeline(values,model)
#pipe
pipe.fit(x_train,y_train)
print(pipe.score(x_train,y_train))
pipe.score(x_test,y_test)
y_pred=pipe.predict(x_test)
print(metrics.r2_score(y_test,y_pred))
print(pipe.predict(pd.DataFrame([['Vistara','Delhi','Hyderabad',0,1,7,21,1,23,3,2,2]],columns=['Airline', 'Source', 'Destination', 'Total_Stops',
       'Journey_Date', 'Journey_Month', 'Dep_Hour', 'Dep_Min', 'Arrival_Hour',
       'Arrival_Min', 'Duration_Hour', 'Duration_Min'])))


# In[73]:


#Using Support Vector Regression Model
from sklearn.svm import SVR
model = SVR(kernel = 'rbf')
pipe=make_pipeline(values,model)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
print(pipe.score(x_train,y_train))
pipe.score(x_test,y_test)
print(metrics.r2_score(y_test,y_pred))
print(pipe.predict(pd.DataFrame([['Vistara','Delhi','Hyderabad',0,1,7,21,1,23,3,2,2]],columns=['Airline', 'Source', 'Destination', 'Total_Stops',
       'Journey_Date', 'Journey_Month', 'Dep_Hour', 'Dep_Min', 'Arrival_Hour',
       'Arrival_Min', 'Duration_Hour', 'Duration_Min'])))


# In[74]:


#Using random forest Regression technique
#Best score
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=30)
pipe=make_pipeline(values,rfr)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
print(pipe.score(x_train,y_train))
print(pipe.score(x_test,y_test))


# In[75]:


print(pipe.predict(pd.DataFrame([['Vistara','Delhi','Hyderabad',0,1,7,21,1,23,3,2,2]],columns=['Airline', 'Source', 'Destination', 'Total_Stops',
       'Journey_Date', 'Journey_Month', 'Dep_Hour', 'Dep_Min', 'Arrival_Hour',
       'Arrival_Min', 'Duration_Hour', 'Duration_Min'])))


# In[ ]:





# In[76]:


#Graph between the Actual and Predicted value of Flight Price
sns.distplot(y_test-pipe.predict(x_test))
a=y_test-pipe.predict(x_test)
a.skew()
plt.savefig('Predicted_Actual.png',format='png',bbox_inches='tight')
plt.show()
#Gaussian distribuation


# In[77]:


plt.scatter(y_test,pipe.predict(x_test))
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.show()


# In[ ]:





# In[78]:


print("MAE:",metrics.mean_absolute_error(y_test,y_pred))
print("MSE:",metrics.mean_squared_error(y_test,y_pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[79]:


#R2 scoce using Random Forest Regressor Technique
metrics.r2_score(y_test,y_pred)


# In[ ]:





# In[80]:


df1.head(2)


# In[81]:


#df3.heAD()


# In[82]:


import pandas as pd
from sklearn import preprocessing

x = df1[['Total_Stops', 'Price',
       'Journey_Date', 'Journey_Month', 'Dep_Hour', 'Dep_Min', 'Arrival_Hour',
       'Arrival_Min', 'Duration_Hour', 'Duration_Min']] #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df3 = pd.DataFrame()
df3['Airline']=df1['Airline']
df3['Source']=df1['Source']
df3['Destination']=df1['Destination']
df4=pd.DataFrame(x_scaled)
df4.columns=x.columns


# In[83]:


df3=pd.concat([df3,df4],axis=1)


# In[84]:


df3.head()


# In[85]:


df3.shape


# In[86]:


df3.dropna(inplace=True)


# In[87]:


#Null Hypothesis->The independent column that is Duration_Hour is not co-related with the target column that is Price
#Alternate Hypothesis->The independent column that is Duration_Hour is co-related with the target column that is Price


# In[88]:


from scipy.stats import pearsonr


# In[89]:


corr1=pearsonr(df4['Price'],df4['Duration_Hour'])
print(corr1)


# Inference-> The p-value is 0.0 an since the p-value(0.0) is less than the level of significance(0.05) we reject the Null Hypothesis and accept the Alternate Hypothesis that is the target column (Price) and the independent column (Duration_Hour) are corelated as peoarson_coefficient is 0.51 which means moderately corelated.This shows that the time duration plays an important role for making the decision to board the flight with best price.

# In[90]:


plt.scatter(df1['Duration_Hour'],df1['Price'])
plt.title("Price vs Duration")
plt.xlabel("Duration (In hours)")
plt.ylabel("Price")
plt.savefig('DurationPrice.png',format='png',bbox_inches='tight')
plt.show()


# In[ ]:





# In[91]:


df4.head()
x=df4.drop('Price',axis=1)
y=df4['Price']


# In[92]:


from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
#pipe=make_pipeline(values,model)
model.fit(x,y)
print(model.feature_importances_)


# In[93]:


feat_imp=pd.Series(model.feature_importances_,index=x.columns)


# In[94]:


feat_imp.nlargest(9).plot(kind='barh')
plt.title("Factors affecting the Price")
plt.ylabel("Factors")
plt.savefig('FactorsPrice.png',format='png',bbox_inches='tight')
plt.show()


# In[95]:


#Null Hypothesis->The independent column that is Total_Stops is not co-related with the target column that is Price
#Alternate Hypothesis->The independent column that is Total_Stops is co-related with the target column that is Price

corr1=pearsonr(df4['Price'],df4['Total_Stops'])
print(corr1)


# Inference-> The p-value is 0.0 an since the p-value(0.0) is less than the level of significance(0.05) we reject the Null Hypothesis and accept the Alternate Hypothesis that is the target column (Price) and the independent column (Total_Stops) are corelated as pearson_coefficient is 0.604 which means moderately corelated.This shows that the time duration plays an important role for making the decision to board the flight with best price.

# In[ ]:





# In[96]:


plt.figure(figsize=(18,18))
sns.heatmap(df1.corr(),annot=True)


# In[97]:


from xgboost import XGBRegressor
model=XGBRegressor()
pipe=make_pipeline(values,rfr)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
print(pipe.score(x_train,y_train))
print(pipe.score(x_test,y_test))


# In[98]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=700,min_samples_split=15,min_samples_leaf=1,max_features="auto",max_depth=20)
pipe=make_pipeline(values,rfr)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
print(pipe.score(x_train,y_train))
print(pipe.score(x_test,y_test))


# In[99]:


metrics.r2_score(y_test,y_pred)


# In[100]:


print("MAE:",metrics.mean_absolute_error(y_test,y_pred))
print("MSE:",metrics.mean_squared_error(y_test,y_pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[101]:


print(pipe.predict(pd.DataFrame([['Vistara','Delhi','Cochin',2,1,7,21,1,23,3,2,2]],columns=['Airline', 'Source', 'Destination', 'Total_Stops',
       'Journey_Date', 'Journey_Month', 'Dep_Hour', 'Dep_Min', 'Arrival_Hour',
       'Arrival_Min', 'Duration_Hour', 'Duration_Min'])))


# In[102]:


#Graph between the Actual and Predicted value of Flight Price
sns.distplot(y_test-pipe.predict(x_test))
a=y_test-pipe.predict(x_test)
a.skew()
#plt.savefig('Predicted_Actual.png',format='png',bbox_inches='tight')
plt.show()
#Gaussian distribuation


# In[103]:


df1.columns


# In[ ]:





# In[104]:


df3.head()


# In[105]:


#Chi-Square test on Source and destination


# In[106]:


crosstab = pd.crosstab(df3["Source"], df3["Destination"])
crosstab


# In[107]:


#Null Hypothesis->Souce and Destination are dependent on each other
#Alternate Hypothesis->Source and destination are independent of each other


# In[108]:


import scipy.stats as stats
stats.chi2_contingency(crosstab)


# Inference->The p value is 0.0 and since the p-value (0.0) of our experiment is less than the level of significance(0.05) we will reject the null hypothesis and accept the alternate hypothesis that is the Source and Destination are independent of each other.

# In[ ]:





# In[109]:


#One-way anova test to check relationship between Journey_Date,Journey_Month and Dep_Hour


# Null Hypothesis->The three columns Journey date,Journey Month and Departure Hour are coreated that is dependent.
# 
# Alternate Hypothesis->The three columns Journey date,Journey Month and Departure Hour are independent of each other.

# In[110]:


#One way Anova
from scipy.stats import f_oneway

def one_way_anova(data1,data2,data3):
    return(f_oneway(data1,data2,data3))

data1=df3['Journey_Date']
data2=df3['Journey_Month']
data3=df3['Dep_Hour']

print(one_way_anova(data1,data2,data3))


# Inference->The p value is 5.39e-90 and since the p-value (5.39e-90) of our experiment is less than the level of significance(0.05) we will reject the null hypothesis and accept the alternate hypothesis that is the three columns Journey date,Journey Month and Departure Hour are independent of each other.

# In[ ]:





# Null Hypothesis->The three columns Arrival Hour,Arrival Min and Duration Hour are co-reated that is dependent.
# 
# Alternate Hypothesis->The three columns Arrival Hour,Arrival Min and Duration Hour are independent of each other.

# In[111]:


#One way Anova
from scipy.stats import f_oneway

def one_way_anova(data1,data2,data3):
    return(f_oneway(data1,data2,data3))

data1=df3['Arrival_Hour']
data2=df3['Arrival_Min']
data3=df3['Duration_Hour']
print(one_way_anova(data1,data2,data3))


# Inference->The p value is 0.0 and since the p-value (0.0) of our experiment is less than the level of significance(0.05) we will reject the null hypothesis and accept the alternate hypothesis that is the three columns Arrival Hour,Arrival Minute and Departure Hour are independent of each other.

# In[ ]:





# Null Hypothesis->The six columns Journey date,Journey Month,Duration Hour, Arrival Hour,Arrival Min and Departure Hour are co-reated that is dependent.
# 
# Alternate Hypothesis->The six columns Journey date,Journey Month,Duration Hour, Arrival Hour,Arrival Min and Departure Hour are independent of each other.

# In[112]:


#One way Anova
from scipy.stats import f_oneway

def one_way_anova(data1,data2,data3,data4,data5,data6):
    return(f_oneway(data1,data2,data3,data4,data5,data6))

data1=df3['Arrival_Hour']
data2=df3['Arrival_Min']
data3=df3['Duration_Hour']

data4=df3['Journey_Date']
data5=df3['Journey_Month']
data6=df3['Dep_Hour']

print(one_way_anova(data1,data2,data3,data4,data5,data6))


# Inference->The p value is 0.0 and since the p-value (0.0) of our experiment is less than the level of significance(0.05) we will reject the null hypothesis and accept the alternate hypothesis that is all the columns Journey date,Journey Month, Duration Hour Arrival Hour,Arrival Minute and Departure Hour are independent of each other.

# In[ ]:





# In[113]:


#Saving this datagrafe into a csv file
df1.to_csv('ModifiedFlightPrice.csv',index=False)


# In[114]:


import pickle
pickle.dump(pipe,open('FlightPricePrediction.pkl','wb'))


# In[ ]:





# In[115]:


sns.pairplot(df1)


# In[ ]:




