# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:15:01 2020

@author: DEEXITH REDDY
"""


##Data has been collected from an ongoing cardiovascular 
##study on residents of the town of Framingham, Massachusetts. 
##The dataset provides the patients’ information. 
##The data has 4238 records, 16 attributes.

##* <b>DEMOGRAPHIC</b> 
##  + Sex, Age
##* <b>BEHAVIORAL</b>
##  + Current smoker, Cigs per day
##* <b>MEDICAL HISTORY</b>
##  + BP Meds, Prevalent Stroke, Prevalent Hyp, Diabetes
##* <b>MEDICAL CURRENT</b>
##  + Tot Chol, Sys BP, Dia BP, BMI, Heart Rate, Glucose
##* <b>TARGET VARIABLE</b>
##  + TenYearCHD - 10 year risk of coronary heart disease)

##We try to determine if there is a casual effect of smoking on the Coronary Heart Disease


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Users/DEEXITH REDDY/Desktop/Projects/Cigarette Smoking Project/framingham.csv")


#Know the values present
df.describe()

#Know find NA values and dropdf.isnull().sum()

df.isnull().sum()
df=df.dropna()

##Splitting data into smoker and non-smoker for effective analysis:

smoker=df.loc[df['currentSmoker']==1]
nonsmoker=df.loc[df['currentSmoker']==0]

##Counting Female and male smokers

smoker['male'].value_counts()

## 1    981 (male)
## 0    807 (Female)


##Visualizing Age
x=smoker['age']
plt.hist(x)
plt.xlabel('AGE')
plt.ylabel('COUNT')
plt.show()

smoker['age'].value_counts()

##We know that majority of smokers are of the age 40

##Visualizing Education
x=smoker['Education']
plt.hist(x)
plt.xlabel('AGE')
plt.ylabel('COUNT')
plt.show()

##Lower education, more prone to smoking

##Renaming Target variable of Ten Year CHD to y

df=df.rename(columns={'TenYearCHD':'y'})


##Correlation plot:

corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

##Straitify split for train and test:

from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.1,stratify=df['y'])


##Cheking correlation:

corr, _ = pearsonr(train['cigsPerDay'], train['y'])
print(corr) ##We come to know that from the correlation, it is very low. We need to add more  variables to know how it improves.

##Import libraries:

import statsmodels.api as sm
from statsmodels.formula.api import logit

formula = 'y ~ cigsPerDay'

model = logit(formula = formula, data = train).fit()

print(model.summary())

##Since the p>|z| value is less greater than 0.05, we need to add more variables

##Pseudo R-Squared=0.026, Deviance=7.3

print((model.null_deviance-model.deviance)/model.null_deviance)

##We see that there is less correlation between age and cigsperday from heatmap. So there will be no problem of multicollinearity. There is also a correlation between age and CHD.
##
formula = ('y ~ cigsPerDay + age')

model = logit(formula = formula, data = train).fit()

print(model.summary())
print((model.null_deviance-model.deviance)
print((model.null_deviance-model.deviance)/model.null_deviance)

##Large Deviance and Loglihood change. Also the coefficient of cigsPerDay increased showing there was downward bias.P-values all below 0.05

##
formula = 'y ~ cigsPerDay + age + male'

model = logit(formula = formula, data = train).fit()

print(model.summary())

print((model.null_deviance-model.deviance)
print((model.null_deviance-model.deviance)/model.null_deviance)
 
##There is little correlation between age, gender and cigsPerDay. Also, some correlation between gender and CHD. 
##Large Deviance and Loglihood change. Also the coefficient of cigsPerDay decreased showing there was upward bias. P-values all below 0.05.
       
##Adding BPmeds to reduce deviance, Pseudo R squared.

formula = 'y ~ cigsPerDay + age + male + BPMeds'

model = logit(formula = formula, data = train).fit()

print(model.summary())

##The coefficient of cigsperDay remains constant.
##There is little correlation between age, gender, BPMeds, cigsPerDay. Also, some correlation between BPMeds and CHD. 
##Large Deviance and Loglihood change. Also the coefficient of cigsPerDay remains constant. P-values all below 0.005.

formula = 'y ~ cigsPerDay + age + male + BPMeds + prevalentStroke'

model = logit(formula, data = train).fit()

print(model.summary())

##Since, the deviance did not change that much and remains the same as 245.58. Also, prevalentStroke addition has p-value of greater than 0.005.

##Now we have to find out the increase in the risk of heart disease for every increase in the cigarette:

AME=model.get_margeff(at='overall',method='dydx')
print(AME.summary())

##CONCLUSION:

##With increase in one cigarette smoking per day, increases the risk of heart diesease by 0.2%







