# Assignment-11.1 - Predicting Car Price

This project aims to build a predictive model to predict car prices in the United States. 

Below illustrates the process and the results of our prediction models.  The 6-step CRISP-DM Process is used in this predictive analytics. 

## Step 1: Business Understanding

Task 1.  Build different predictive model using Linear Regression by varying the degree of PolynomialFeatures, LASSO model to determine top factors and Ridge Model.

Measure of effectiveness:

1. Mean Squared Error of the Training datasets

2. Mean Squared Error of the Training datasets

3. Explained variance to determine how much of the variance can be explained by the model

Task 2. Using the above metrics, determine the best model 

Task 3. For the best model, present the coefficients


## Step 2: Data Understanding 
### Some steps taken to get to know the dataset and identify any quality issues within. By taking time to get to know the dataset and explore what information it contains and how this could be used to inform business understanding.

a.	I obtained a set of car price data and the data fields are found in the figure below:
![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/5ead5be5-a481-4d60-857a-a24cd44342f3)

b.	I did a quick plot of the car prices to determine if the data is sensible.  Because of outliers, it was impossible to study the data visually.  
![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/78528cd2-b301-4d1c-912d-a321b69bfb51)

c. I did a quick search online and run different values before deciding to limit the car prices to no more than $250,000.  
![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/99ca64fe-8ec6-4167-a7f2-67fba69efae6)
![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/c248cd67-6842-401d-972c-4b19532bb902)

d.	I studied each data field in turn and eliminate the factors that will affect car prices e.g. id and VIN.  I did a quick google and it appears that VIN is a series of alphanumeric data that characterised the car but most of the important features were already captured in the other data fields.

e.	Many of the factors were captured as categorical data, e.g. condition = {good, excellent, like new, fair, new, salvage} and needed to be changed into numerical factor. 

f.	Because some categorical data has information already captured in other factors, e.g. region & state, I decided to keep the information on state and discard region to limit the number of factors 

g.	Most importantly, in order to ensure the predicted car price would always be positive, I decided to eliminate all car with prices = 0 and apply logarithm on car prices for modelling. 



## Step 3: Data Preparation
### Based on the considerations stated out above under Data Understanding, data cleansing process was initiated.  

a. Digitise the categorical data using pd.get_dummies (Note that "paint_color" and "state" was later dropped from the list of useful data after running the model and deciding to drop the number of features)
![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/09145438-d8a1-4fef-ab17-7fd6edcf219f)

b. Remove all the unnecessary data that we did not intend to train the model with (Note that "paint_color" and "state" was later dropped from the list of useful data after running the model and deciding to drop the number of features)
![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/abc1386a-5405-448a-bbc3-bda45834256b)

c.	Limit the value of car prices and apply logarithmic function to the car price.  

  i. As above, I limited the car prices to at most $250,000 because car prices greater than $250,000 do not sound realistic and even if it were included, they are often outliers and can skewed the data and predictions.  

  ii. When I first run the model, many of the price predictions were less than zero.  Hence, I decided to apply a logarithmic model on the car prices so that after applying an exponential function to reverse the logarithm, the price is positive.  Also, logarithm cannot be applied on value zero.

![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/8a61b620-4bc5-44f7-a18b-1da1b725f27f)

d. Because there are a whopping 92 features to train, I then run the correlation table to see the correlations of price to factors (even though this only measures linear relationship).  As manufacturer's information only first appears at the 15th position, I thought of training 2 models, one including manufacturer, the other excluding and determine if there is a huge difference in the model.  

![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/3930d3b7-6b80-4c23-af17-0466d4801f59)


e. Split the data into training and development sets at a split of 70-30.  



## Step 4: Modelling 
### 5 models were deployed

a. Model 1: Ordinary linear regression with 92 features

The training MSE was 1.1806, while the development MSE was 1.1719.  The predicted and observed vaues are plotted below.  Interestingly, the observed values appear to be greater than the predictions.
![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/4cc218fb-86de-4b04-a7bf-667866524302)


b. Model 2: Ordinary linear regression with 50 features

The training MSE was 1.1961, while the development MSE was 1.2025.  The predicted and observed vaues are plotted below.  Again, the observed values appear to be greater than the predictions.
![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/1758beca-fca1-4e6e-8a3a-9104873cb783)



c. Model 3: Linear regression model on PolynomialFeatures with degree = 2 with 50 features

While the training MSE was only 0.9175, the development MSE was 1.3094e+23.  The predicted and observed vaues are plotted below.  It appears that the prediction model made a few predictions that well-exceeded the reasonable range for car prices.
![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/60e7169f-966f-44dd-ade8-4b8bc2e1e387)


d. Model 4: LASSO model 

I apply a degree = 2 PolynomialFeature on the LASSO model.   The training MSE was 1.5672 while the development MSE was 1.5817.  The results are plotted below.  

![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/3022541b-dea5-4549-a5d6-e76f5da554ae)

Because the predicted values look odd, I looked into the lasso coefficients and realise that almost all the coefficients were zero and the model is almost useless. 

![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/55cb7e9e-7f36-45f3-bf05-77cb3bae2a4c)

e. Model 5: Applying ridge model 

I applied PolynomialFeatures for degree = 2 and applied GridSearchCV to alpha value = {0.1, 1 and 10}.  The best alpha value was 1 and the predicted values are shown below.  The training MSE was 0.9197 and the development MSE was 0.9643. 
![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/ae38443f-d70a-410d-b612-413884d8f18b)


## Step 5: Evaluation
### A metric table was developed to evaluate the models. 

A dataframe containing the training error (train_mse), development error (test_mse) and the explained variance is shown below.  The model with the lowest training error was Model 3, the  the linear regression model trained on degree = 2  but its development error was way too high at 1.73e+21.  The model with the lowest development error was Model 5 - Ridge model and it also has a relatively low training error.  The model also has one of the highest explained variance of 0.4143.  Hence, we will proceed to deploy Model 5 - ridge model with alpha = 1 and Polynomial Features(degree = 2). 

![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/04534e5f-6ed6-4336-a874-57c42cab7d11)


## Step 6: Deployment 

Model 5 - ridge model with alpha = 1 and Polynomial Features (degree = 2) will be deployed as it is the best performing model.  Here, I sorted the features by the magnitude of its coefficients and found that similar to all models, the "year" and "year^2" features have the highest coefficients. In fact, features which are multiples of the feature "year" appears to have the highest coefficients.  Hence, it is important that this feature be included in the data collected in the future.  

![image](https://github.com/CarolTeo11/Assignment-11.1/assets/130137674/05310859-2f83-44a1-bb3b-8e999f1fb58b)

However, if the client has new data, he/she can run the model using np.exp(best_model.predict(new_data)) to estimate the price of the car. 
