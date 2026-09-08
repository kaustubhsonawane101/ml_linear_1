def train_model():
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

    '''PREDICT'''
    y_predict=model.predict(x_test_scaled)

    from sklearn.metrics import r2_score 
    r2=r2_score(y_test,y_predict) 
    return r2

if __name__== "__main__":
    r2=train_model()
    print(f'r2:{r2}')

#in this way our model is under the function, and it can print the result here itself, and it can transported to test file also.
#here __main__ means present file, not a file name