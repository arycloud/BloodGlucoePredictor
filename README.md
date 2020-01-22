# BloodGlucoePredictor

In this project, we have to predict the Blood Glucose level for an hour in the future at every 5 minutes interval. 

#### -- Project Status: [Completed]

## Project Intro/Objective
The purpose of this project is to predict the Blood Glucose level.

### Methods Used
* Machine Learning
* Data Visualization
* Predictive Modeling
* Deep Learning
* Auto Regressor, ARIMA & LSTM 

### Technologies
* Python
* Pandas, jupyter, Numpy, Keras, Tensorflow

## Project Description
I have tackled this problem from a Time Series perspective and tried different machine learning methods. At the very first glance,I have tried Auto-regressor & ARIMA models but after digging deeply inside the available datasets and realize the shortage of feature set and week correlations b/w different features I decided to use Deep Learning and specifically choose LSTM, which gave me the state-of-the-art results for this model.

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [/data](Repo folder containing raw data) within this repo.
    
3. Data processing/transformation script is being kept [/src/data/preprocess.py]
4. Model implementation functions are kept at [src/models/*]
5. Clark Error Grid implementation can found at [src/visualization]
6. The trained and saved model can be found at [/models]
7. Some jupyter notebooks for analysis can be found at [/notebooks/]
8. Plot images can be found at [/plot_images/]
9 All requirements are added to the [requirements.txt] file.



## Contributing DSWG Members

**Team Leads (Contacts) : [AbdulRehman](https://github.com/arycloud)(@Abdul)**

#### Other Members:

|Name     |  Slack Handle   | 
|---------|-----------------|
|[Abdul Rehman](https://github.com/arycloud)| @Abdul       |

## Important Note:
I have achieved the required result(accuracy with RMSE & Clark Error Grid), but I believe I can improve this model a lot if I got enough time.

## Contact
* You can contact me at abdul@pythonest.org OR abdul12391@gmail.com  

##
