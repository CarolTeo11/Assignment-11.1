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

a. 



## Modelling 


## Evaluation


## Deployment 
