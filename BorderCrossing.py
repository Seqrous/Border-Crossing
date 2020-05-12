# author: Seqrous
# date: 11.05.2020
# kaggle: https://www.kaggle.com/divyansh22/us-border-crossing-data

# standard imports 
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import calendar
import seaborn as sns
# forecasting imports
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM

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

regression_data = dataset[['Value', 'Month', 'Year']].groupby('Month')['Value'].sum()