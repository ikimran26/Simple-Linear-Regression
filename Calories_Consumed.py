

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

wcat=pd.read_csv("C:\\Users\\mozak\\Desktop\\ExcelR\\Assigments\\Simple Linear Regression\\Python\\calories_consumed.csv")
plt.scatter(x=wcat.Calories_Consumed,y=wcat.Weight_gained,color='green')
help(np.corrcoef)
wcat.describe()
import statsmodels.formula.api as smf

model = smf.ols('Weight_gained~Calories_Consumed',data=wcat).fit()
model.params
model
model.summary()

pred=model.predict(wcat.iloc[:,0])

print(model.conf_int(0.01))# cnfudence intervaal of 99%
res=wcat.Weight_gained-pred
res

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(wcat.Weight_gained, pred))
rmse


-------------------------------------------
####x= log(Calories_Consumed)   ,  y  = Weight_gained######

model2= smf.ols('Weight_gained ~ np.log(Calories_Consumed)',data=wcat).fit()
model2.params
model2

pred2=model2.predict(wcat.iloc[:,0])  # wcat.iloc[:,0] means Waist

model2.summary()
print(model2.conf_int(0.01))
res=wcat.Weight_gained-pred2

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(wcat.Weight_gained, pred2))
rmse   #  rmse= 32.49688490932126
####x= Waist   ,  y  = log(AT)######
model3=smf.ols('np.log(Weight_gained) ~ Calories_Consumed',data=wcat).fit()
model3.params
model3.summary()
pred_log=model3.predict(pd.DataFrame(wcat.Waist))
pred3=np.exp(pred_log)
pred3=model3.predict() 

pred3.corr(wcat.Waist)

np.corrcoef(wcat.Waist,wcat.AT)
model3.summary()
print(model2.conf_int(0.01))
res=wcat.AT-pred2
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse3=sqrt(mean_squared_error(wcat.AT,pred3))
-------------
------------------------

exp polynomial model
 
----x= Waist*Waist   y=log(AT)----       
Waist_Sq = wcat.Waist*wcat.Waist

model4= smf.ols("np.log(AT) ~ Waist+Waist_Sq",data=wcat).fit()

model4.params
model4.summary()
model.conf_int(0.05)
pred4=model4.predict(wcat.Waist)

from pydoc import help
from scipy.stats.stats import pearsonr
help(pearsonr)

>>>
Help on function pearsonr in module scipy.stats.stats:

pearsonr(wcat.AT,pred4)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse4=sqrt(mean_squared_error(wcat.AT,pred4))
rmse4
-----------------------------------------------------------------