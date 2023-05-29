# Assignment-11.1 - Predicting Car Price

This project aims to build a predictive model to predict car prices in the United States. 

Below illustrates the process and the results of our prediction models.  The CRISP-DM Process is used in this predictive analytics. 

## Business Understanding
Task 1.  Build a predictive model using Linear Regression and determine which factors have the highest correlation and how much each factor contribute to explained variance.

Task 2. Determine the top 5 factors affecting car price

Task 3. Can the price of a car be predicted based on its attribute with reasonable accuracy?


## Data Understanding 
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



## Data Preparation
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



## Modelling 


## Evaluation


## Deployment 
