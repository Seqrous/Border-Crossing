# author: Seqrous
# date: 11.05.2020
# kaggle: https://www.kaggle.com/divyansh22/us-border-crossing-data

# standard imports 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import calendar
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# forecasting imports
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Model

# laod the dataset
dataset = pd.read_csv('data/Border_Crossing_Entry_Data.csv')
print(dataset.head())

# check for missing values
dataset.isna().sum().any()

# convert date to: month and year columns
dataset['Month'] = dataset['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M').month).astype(int)
dataset['Month'] = dataset['Month'].apply(lambda x: calendar.month_abbr[x])
dataset['Year'] = dataset['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M').year)

""" 1. plot how the amount was changing over time """
# I do know that the similar graph is on dataset's page in data overview section

year_value_data = dataset.groupby('Year')['Value'].sum()

plt.figure(figsize=(12, 8))
plt.bar(year_value_data.index, year_value_data.values)
plt.xlabel('Year', fontsize=15)
plt.xticks(year_value_data.index, rotation=45)
plt.ylabel('Value', fontsize=15)
plt.title('History of the amount of all border crossing incidents 1996-2020', fontsize=15)
plt.show()

""" 2. Differentiate between Mexico and Canada """

year_value_Mexico = dataset[dataset['Border'] == 'US-Mexico Border'].groupby('Year')['Value'].sum()
year_value_Canada = dataset[dataset['Border'] == 'US-Canada Border'].groupby('Year')['Value'].sum()

year_val_Mexico = pd.DataFrame({
            'Year': year_value_Mexico.index,
            'Country': 'Mexico',
            'Count': year_value_Mexico.values
        })

year_val_Canada = pd.DataFrame({
            'Year': year_value_Canada.index,
            'Country': 'Canada',
            'Count': year_value_Canada.values
        })

year_val_concat = pd.concat([year_val_Mexico, year_val_Canada], ignore_index=True)

# show the plot 
plt.figure(figsize = (12, 8))
sns.barplot(x='Year', y='Count', hue='Country', data=year_val_concat)
plt.title('Border crossing each year - Mexico / Canada ', fontsize=15)
plt.xlabel('Year', fontsize=15)
plt.xticks(rotation=45)
plt.ylabel('Number of incidents', fontsize=15)
plt.legend()
plt.show()

""" 3. Average month crossing distribution over all years """

# mean for each month
mean_month_val = dataset.groupby('Month')['Value'].mean()

# sort in a month order
ordered_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
mean_month_val.index = pd.Categorical(mean_month_val.index, 
                   categories = ordered_months, ordered = True)
mean_month_val = mean_month_val.sort_index()

# show the plot
plt.figure(figsize=(12,8))
sns.set(style='whitegrid')
sns.lineplot(data=mean_month_val)
plt.xlabel('Month', fontsize=15)
plt.ylabel('Border crossing amount', fontsize=15)
plt.title('Amount of border crossing for each month on average', fontsize=15)
plt.show()

""" 4. Most popular way of transport over the years for Canada/Mexico """

# Mexico most popular
transport_data_Mexico = dataset[dataset['Border'] == 'US-Mexico Border']

transport_Mexico_overview = transport_data_Mexico.groupby('Measure')['Value'].sum()
transport_Mexico_overview.sort_values(ascending=False, inplace=True)

print('Most popular transport on Mexican border')
for i, (transport, value) in enumerate(zip(transport_Mexico_overview.index, transport_Mexico_overview)):
    print(f"{i+1}. {transport}: {value}")

# line plot, a line for each transport
plt.figure(figsize=(12, 8))
sns.lineplot(x='Year', y='Value', hue='Measure', data=transport_data_Mexico)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Border crossing amount', fontsize=15)
plt.title('Most popular transport on Mexico border', fontsize=15)
plt.show()


# Mexico less popular
transport_data_Mexico_low = transport_data_Mexico[(transport_data_Mexico['Measure'] != 'Personal Vehicle Passengers') &
                                                  (transport_data_Mexico['Measure'] != 'Personal Vehicles') &
                                                  (transport_data_Mexico['Measure'] != 'Pedestrians')]

plt.figure(figsize=(12, 8))
sns.lineplot(x='Year', y='Value', hue='Measure', data=transport_data_Mexico_low)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Border crossing amount', fontsize=15)
plt.title('Less popular transport on Mexico border', fontsize=15)
plt.show()                                           
                                                  
# Canada most popular
transport_data_Canada = dataset[dataset['Border'] == 'US-Canada Border']

transport_Canada_overview = transport_data_Canada.groupby('Measure')['Value'].sum()
transport_Canada_overview.sort_values(ascending=False, inplace=True)                                                  
                              
print('Most popular transport on Canadian border')
for i, (transport, value) in enumerate(zip(transport_Canada_overview.index, transport_Canada_overview)):
    print(f"{i+1}. {transport}: {value}")                    
# Pedestrians seem to be less common in Canada
                                             
plt.figure(figsize=(12, 8))
sns.lineplot(x='Year', y='Value', hue='Measure', data=transport_data_Canada)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Border crossing amount', fontsize=15)
plt.title('Most popular transport on Canadian border', fontsize=15)
plt.show()     
                                                  
# Canada less popular
transport_data_Canada_low = transport_data_Canada[(transport_data_Canada['Measure'] != 'Personal Vehicle Passengers') &
                                                  (transport_data_Canada['Measure'] != 'Personal Vehicles')]
                                                  
plt.figure(figsize=(12, 8))
sns.lineplot(x='Year', y='Value', hue='Measure', data=transport_data_Canada_low)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Border crossing amount', fontsize=15)
plt.title('Less popular transport on Canadian border', fontsize=15)
plt.show()                                                 
                                                  
""" 5. Boxplot for each state and people arriving in it """
# checking most popular states, so later they can be divided into popular and unpopular
print(dataset.groupby('State')['Value'].sum().sort_values(ascending=False))

# popular states
popular_state_data = pd.DataFrame()
popular_states = ['TX', 'CA', 'AZ', 'NY', 'MI', 'WA', 'ME']
for state in popular_states:
    current_state = pd.DataFrame(dataset[dataset['State'] == state].groupby('Year')['Value'].sum())
    current_state['State'] = state
    current_state['Median'] = current_state['Value'].median()
    popular_state_data = pd.concat([popular_state_data, current_state], ignore_index=True)

plt.figure(figsize = (19, 10))
sns.boxplot(x = 'Value', y = 'State', data=popular_state_data, fliersize = 0)
plt.title('Number of people arriving at the popular states', fontsize = 15)
plt.xlabel('Number of people', fontsize = 15)
plt.ylabel('State', fontsize = 15)
plt.xticks(popular_state_data['Median'], rotation=45)
plt.grid()
plt.show()

# unpopular states
less_popular_state_data = pd.DataFrame()
less_popular_states = ['VT', 'MN', 'ND', 'NM', 'MT', 'ID', 'AK', 'OH']
for state in less_popular_states:
    current_state = pd.DataFrame(dataset[dataset['State'] == state].groupby('Year')['Value'].sum())
    current_state['State'] = state
    current_state['Median'] = current_state['Value'].median()
    less_popular_state_data = pd.concat([less_popular_state_data, current_state], ignore_index=True)
    
plt.figure(figsize = (19, 10))
sns.boxplot(x = 'Value', y = 'State', data=less_popular_state_data, fliersize = 0)
plt.title('Number of people arriving at the unpopular states', fontsize = 15)
plt.xlabel('Number of poeple', fontsize = 15)
plt.ylabel('State', fontsize = 15)
plt.xticks(less_popular_state_data['Median'], rotation=45)
plt.grid()
plt.show()

""" Forecasting for 2020 using Keras' LSTM """

# get the necessary data
data = dataset[['Value', 'Month', 'Year']]
years = list(data['Year'].value_counts().index)
years.sort()

# for each month in year calculate the total sum of border crossings
# to the sum append its year and month
feed_data = pd.DataFrame()
for year in years:
    data_each_year = data[data['Year'] == year]
    month_in_year = pd.DataFrame(data_each_year.groupby('Month')['Value'].sum())
    month_in_year['Month'] = month_in_year.index
    month_in_year['Year'] = year
    feed_data = pd.concat([feed_data, month_in_year], ignore_index=True)

# convert month abbreviation to numerical value
def month_converter(month):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return months.index(month) + 1

# sort by year, month
feed_data['Month'] = feed_data['Month'].apply(lambda x: month_converter(x))
feed_data = feed_data.sort_values(['Year', 'Month'], ascending=['True', 'True'])
feed_data = feed_data.reset_index()

# get necessary data
feed_data_value = feed_data['Value']
feed_data_scaled = np.array(feed_data_value).reshape(-1, 1)

# Scaling
scaler = MinMaxScaler()
feed_data_scaled = scaler.fit_transform(feed_data_scaled)

# split into test and train sets
train, test = train_test_split(feed_data_scaled, test_size=0.3, shuffle=False)
# convert an array of values into a dataset matrix
def array_to_matrix_with_shift(dataset, look_back=1):
    data_X = []
    data_y = []
    for i in range(len(dataset)-look_back):
        element = dataset[i:(i+look_back), 0]
        data_X.append(element)
        data_y.append(dataset[i+look_back, 0])
    return np.array(data_X), np.array(data_y)

look_back = 1
train_X, train_y = array_to_matrix_with_shift(np.array(train).reshape(-1, 1))
test_X, test_y = array_to_matrix_with_shift(np.array(test).reshape(-1, 1))

# LSTM expects 3 inputs
# samples 
# time steps - one time step is one observation (so Value?)
# features - month and  year?
# all in all, (sample, time_step, (feature1, feature2, ...))

train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

# train_X - Jan 1996 - Oct 2012
# test_X - Dec 2012 - Feb 2020

""" Training the model """
# neural network using LSTM
model = Sequential()
model.add(LSTM(20, input_shape=(1, look_back)))
model.add(Dense(1))

model.compile(optimizer='adam',
              loss='mae')

r = model.fit(train_X, train_y, epochs=40, batch_size=5, validation_data=(test_X, test_y))

# plot history
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

""" Prediction/Evaluation """
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# months corresponding to predicted values
month_idx_start = np.array([[12]]) # wraps up 2012
month_idx_mid = np.arange(1, 13) # returns 1-12
# months for 2013, 2014, 2015, 2016, 2017, 2018, 2019
month_idx_mid = np.tile(month_idx_mid, 7).reshape(-1, 1)
month_idx_end = np.array([[1]]) # wraps up 2020 (there's only Jan in 2020)
full_months = np.concatenate((month_idx_start, month_idx_mid, month_idx_end))

# prepare inverted y data
inv_y_df = pd.DataFrame(inv_y, columns=['Value'])
inv_y_df['Month'] = full_months
inv_y_df['Month'] = inv_y_df['Month'].astype(int)
inv_y_df['Month'] = inv_y_df['Month'].apply(lambda x: calendar.month_abbr[x])
inv_y_df['Type'] = 'Actual'

# prepare inverted yhat data
inv_yhat_df = pd.DataFrame(inv_yhat, columns=['Value'])
inv_yhat_df['Month'] = full_months
inv_yhat_df['Month'] = inv_yhat_df['Month'].astype(int)
inv_yhat_df['Month'] = inv_yhat_df['Month'].apply(lambda x: calendar.month_abbr[x])
inv_yhat_df['Type'] = 'Predicted'

full_results_df = pd.concat([inv_y_df, inv_yhat_df])

start_years = np.array([2012])
# append years, that is 2013 - 2019 (12 times each) + 2 times 2020
mid_years = np.array([2013, 2014, 2015, 2016, 2017, 2018, 2019])
mid_years = np.repeat(mid_years, 12)
last_years = np.array([2020])
years = []
years = np.append(start_years, mid_years)
years = np.append(years, last_years)
years = np.tile(years, 2)

full_results_df['Year'] = years

full_results_df['Date'] = full_results_df['Month'] + full_results_df['Year'].astype(str)
ordered_dates = full_results_df['Date'].unique()
full_results_df['Date'] = pd.Categorical(full_results_df['Date'], categories=ordered_dates,
               ordered=True)

plt.figure(figsize=(25, 8))
sns.lineplot(x='Date', y='Value', hue='Type', data=full_results_df)
plt.legend()
plt.grid(True)
plt.xticks(rotation=90)
plt.show()
