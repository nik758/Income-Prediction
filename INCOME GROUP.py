import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#IMPORT DATA
data = pd.read_csv(r'C:\Users\nikhi\Desktop\project\COMPLETE PROJECT\analytics vidya\income group experiment with data\train.csv')

#==============================================================================
                           #UNIVARIATE ANALYSIS
data.dtypes

#1. CONTINUOUS VARIABLES:
data.describe()

#2. CATEGORICAL VARIABLES:
categorical_variables = data.dtypes.loc[data.dtypes == 'object'].index
print(categorical_variables)

data[categorical_variables].apply(lambda x: len(x.unique()))

#2.1 Analyzing Race variable
data['Race'].value_counts()
data['Race'].value_counts() / data.shape[0]

#2.2 Analyzing Native-Country
data['Native.Country'].value_counts()
data['Native.Country'].value_counts() / data.shape[0]

#==============================================================================
                          #Multivariate Analysis

#1.Both Categorical variable:

#lets takes example of sex and income group
#print the cross-tabulation
ct = pd.crosstab(data['Sex'], data['Income.Group'], margins = True)
print(ct)

#we can also plot it using a stacked chart
%matplotlib inline
ct.iloc[:-1,:-1].plot(kind='bar', stacked=True, color= ['red','blue'],grid=False)


'''Though important, absolute numbers might not be very intuitive to interpret.
   Next lets try to plot the percentage of females and males in each income group.'''
   
def percConvert(sex):
    return sex/float(sex[-1])
   
ct2 = ct.apply(percConvert, axis=1)
ct2
ct2.iloc[:-1,:-1].plot(kind='bar', stacked=True, color= ['red','blue'],grid=False)

#=========================
#2. BOTH CONTINUOUS

data.plot('Age', 'Hours.Per.Week', kind='scatter')
#===================

#3. CATEGORICAL-CONTINUOUS COMBINATION

data.boxplot(column = 'Hours.Per.Week', by='Sex')
#==============================================================================
                         #MISSING VALUE TREATMENT

#1. Checking missing values
# Checking missing values in train data and test data
data.apply(lambda x: sum(x.isnull()))            #or X.isnull().sum()

#2. Imputation
     #since they are categorical variable so impute with mode

#Import function:
from scipy.stats import mode

#try it out
mode(data['Workclass']).mode[0]

#impute the values
var_to_impute = ['Workclass','Occupation','Native.Country']
for var in var_to_impute:
    data[var].fillna(mode(data[var]).mode[0], inplace =True)

#now check missing value
data.apply(lambda x: sum(x.isnull()))            #or X.isnull().sum()

#=============================================================================
                          #OUTLIER TREATMENT

'''We can check outliers in numerical variables by creating simple scatter plots.
   Lets do it for both the numerical variables.'''

%matplotlib inline
#making scatter plot for age
data.plot('ID','Age',kind='scatter')
#making scatter plot for Hours.Per.Week
data.plot('ID','Hours.Per.Week',kind='scatter')

#==============================================================================
                        #Variable Transformation

#check the dtype of variable
data.dtypes    

#Workclass Example   :-
                 
#let takes woorkclass variable as an example

#dteremine the percentage of observation in each category
data['Workclass'].value_counts()/data.shape[0]

'''Depending on the business scenario, we can combine the categories with very few observations. As a thumbrule, lets combine categories with 
   less than 5% of the values.'''
   
category_to_combine = ['State-gov','Self-emp-inc','Federal-gov','Without-pay','Never-worked']

#run a loop and replace all values with others
for  cat in category_to_combine:
    data['Workclass'].replace({cat:'Others'},inplace = True)
    
#check the new categories in workclass
data['Workclass'].value_counts()/data.shape[0]

#COMBINING THE REST:

#STEP1: MAKE A LIST OF VARIABLES TO COMBINE
#maek list of categorical variable:
categorical_variable = list(data.dtypes.loc[data.dtypes == 'object'].index)
categorical_variable

#remove workplace
categorical_variable = categorical_variable[1:]

#check the  current unmber of unique values
data[categorical_variable].apply(lambda x: len(x.unique()))

#STEP2: RUN A LOOP OVER THESE VALUES AND COMBINE CATEGORIES

for column in categorical_variable:
    #Determine the categories to combine:
    frq = data[column].value_counts()/data.shape[0]
    categories_to_combine = frq.loc[frq.values < 0.05].index
    
#loop over all categories and combine them as others
for  cat in categories_to_combine:
    data[column].replace({cat:'Others'},inplace = True)
    
#checking the result for train data
data[categorical_variable].apply(lambda x: len(x.unique()))
        
#==============================================================================
                     #Predictive Modeling
    
#Step1: Data Preprocessing

from sklearn.preprocessing import LabelEncoder   

categorical_variable = data.dtypes.loc[data.dtypes == 'object'].index
categorical_variable

#now we convert them using labelencoder
le = LabelEncoder()

for var in categorical_variable:
    data[var] = le.fit_transform(data[var])
    
#check using dtype
data.dtypes

#===================
#Step2: Fit the model
from sklearn.tree import DecisionTreeClassifier

dependent_variable = 'Income.Group'
independent_variable = [x for x in data.columns if x not in[dependent_variable]]
print (independent_variable)

'''Now that we have the predictors, lets run the model with the following benchmark parameters:

max_depth = 10
min_samples_leaf = 100
max_features = 'sqrt'         '''

#intialize algorithm
model = DecisionTreeClassifier(max_depth = 10,min_samples_leaf = 100,
                               max_features = 'sqrt' )

#fit the algorithm
model.fit(I_train[independent_variable],I_train[dependent_variable])

#Step3: Make predictions
#Now we will use the predict function to make predictions
prediction_I_train = model.predict(I_train[independent_variable])
prediction_I_test = model.predict(I_test[independent_variable])


#Step4: Analyze results:
from sklearn.metrics import accuracy_score

#determine the X_error
acc_I_train = accuracy_score(I_train[dependent_variable],prediction_I_train)
acc_I_train

print'Train Accuracy: %f'  % acc_I_train















































