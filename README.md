# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries and load the dataset.
2. Preprocess and scale the input and output data.
3. Train the Linear Regression model using Gradient Descent.
4. Predict the profit for new city data and display the result.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: P.Dharshini
RegisterNumber: 212225040071
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X = np.c_[np.ones((len(X1), 1)), X1]
    theta = np.zeros((X.shape[1], 1))
    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y

        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta
data = pd.read_csv("50_Startups.csv")
print(data.head())
print("\n")
X = data.iloc[:, :-1].values
from sklearn.preprocessing import LabelEncoder
if isinstance(X[0][-1], str):
    le = LabelEncoder()
    X[:, -1] = le.fit_transform(X[:, -1])
X1 = X.astype(float)
y = data.iloc[:, -1].values.reshape(-1, 1)
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X1_scaled = x_scaler.fit_transform(X1)
y_scaled = y_scaler.fit_transform(y)
print(X1_scaled)
print("\n")
theta = linear_regression(X1_scaled, y_scaled)
new_data = np.array([[165349.2, 136897.8, 471784.1, 2]])
new_scaled = x_scaler.transform(new_data)
new_scaled = np.c_[np.ones((1, 1)), new_scaled]
prediction_scaled = new_scaled.dot(theta)
prediction = y_scaler.inverse_transform(prediction_scaled)
print("Scaled Prediction:", prediction_scaled)
print("Predicted value:", prediction)
```

## Output:
![linear regression using gradient descent](sam.png)
<img width="1920" height="1080" alt="Screenshot (618)" src="https://github.com/user-attachments/assets/2feb71e9-687f-4d4b-89e4-92315b7f32c6" />
<img width="1920" height="1080" alt="Screenshot (617)" src="https://github.com/user-attachments/assets/7226441f-18b2-4572-a4c6-1469102945f8" />
<img width="1920" height="1080" alt="Screenshot (616)" src="https://github.com/user-attachments/assets/fae9c1b1-9660-4d60-88e9-1f59dca849c2" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
