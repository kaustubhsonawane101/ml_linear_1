from sklearn.datasets import fetch_california_housing
data=fetch_california_housing(as_frame=True) 
df=data.frame 
x=df.drop('MedHouseVal',axis=1)
y=df.MedHouseVal

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train_scaled=scaler.transform(x_train)  
x_test_scaled=scaler.transform(x_test)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train_scaled,y_train)  
print(f'coefficients :{model.coef_}')
print(f'intercept:{model.intercept_}')
'''PREDICT'''
y_predict=model.predict(x_test_scaled)
print(f'prediction:{y_predict}')

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
mse=mean_squared_error(y_test,y_predict)  
mae=mean_absolute_error(y_test,y_predict)   
r2=r2_score(y_test,y_predict) 

print(f'mse:{mse},mae:{mae},r2:{r2}')