
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

empdata=pd.read_csv("C:\\Users\\mozak\\Desktop\\ExcelR\\Assigments\\Simple Linear Regression\\Python\\emp_data.csv")

plt.scatter(x=empdata.Salary_hike,y=empdata.Churn_out_rate,color='red')
help(np.corrcoef)

import statsmodels.formula.api as smf

model = smf.ols('Churn_out_rate~Salary_hike',data=empdata).fit()
model.params
model
model.summary()

pred=model.predict(empdata.iloc[:,0])

print(model.conf_int(0.01))# cnfudence intervaal of 99%
res=empdata.Churn_out_rate-pred
res

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(empdata.Churn_out_rate, pred))
rmse


-------------------------------------------


model2= smf.ols('Churn_out_rate ~ np.log(Salary_hike)',data=empdata).fit()
model2.params
model2

pred2=model2.predict(empdata.iloc[:,0])  # wcat.iloc[:,0] means Waist

model2.summary()
print(model2.conf_int(0.01))
res=empdata.Churn_out_rate-pred2

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(empdata.Churn_out_rate, pred2))
rmse   #  rmse= 32.49688490932126
####x= Waist   ,  y  = log(AT)######
model3=smf.ols('np.log(Churn_out_rate)~Salary_hike',data=empdata).fit()
model3.params
model3.summary()
pred_log=model3.predict(pd.DataFrame(empdata.Salary_hike))
pred3=np.exp(pred_log)
pred3=model3.predict() 



np.corrcoef(empdata.Salary_hike,pred3)

print(model2.conf_int(0.01))
res=empdata.Churn_out_rate-pred3
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse3=sqrt(mean_squared_error(wcat.AT,pred3))
-------------
------------------------

exp polynomial model
 
----x=Salary_hike    y=log(churn_out_rate)----       
Salary_Sq = empdata.Salary_hike*empdata.Salary_hike

model4= smf.ols("np.log(Churn_out_rate) ~ Salary_hike+Salary_Sq",data=empdata).fit()

model4.params
model4.summary()
model.conf_int(0.05)
pred4=model4.predict(empdata.Salary_hike)

from pydoc import help
from scipy.stats.stats import pearsonr
help(pearsonr)

>>>
Help on function pearsonr in module scipy.stats.stats:

pearsonr(empdata.Churn_out_rate,pred4)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse4=sqrt(mean_squared_error(empdata.Churn_out_rate,pred4))
rmse4
-----------------------------------------------------------------
