import pandas as pd
import numpy as np
import math
from datetime import datetime
from sklearn.linear_model import LinearRegression
import streamlit as st

df=pd.read_csv('ipl.csv')

df=df.drop(labels=['mid','batsman','bowler','striker','non-striker'],axis=1)

consistent_team=['Kolkata Knight Riders','Chennai Super Kings','Rajasthan Royals','Mumbai Indians',
                  'Kings XI Punjab','Royal Challengers Bangalore','Delhi Daredevils',
                  'Sunrisers Hyderabad']

## 2021 teams

df=df[df['bat_team'].isin(consistent_team) & df['bowl_team'].isin(consistent_team)]

df['date']=df['date'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d') )
bat_team_dict={'Kolkata Knight Riders':1,'Chennai Super Kings':2,'Rajasthan Royals':3,'Mumbai Indians':4,
                  'Kings XI Punjab':5,'Royal Challengers Bangalore':6,'Delhi Daredevils':7,
                  'Sunrisers Hyderabad':8}
bowl_team_dict={'Kolkata Knight Riders':1,'Chennai Super Kings':2,'Rajasthan Royals':3,'Mumbai Indians':4,
                  'Kings XI Punjab':5,'Royal Challengers Bangalore':6,'Delhi Daredevils':7,
                  'Sunrisers Hyderabad':8}
df['bat_team_encoded']=df.bat_team.map(bat_team_dict)
df['bowl_team_encoded']=df.bowl_team.map(bowl_team_dict)

df=df.drop(labels=['venue','bat_team','bowl_team'],axis=1)
df=df[['bat_team_encoded','bowl_team_encoded','overs','wickets','runs','runs_last_5','wickets_last_5','date','total']]

## splitting data into train and test

x_train=df.drop(['total'],axis=1)[df['date'].dt.year<=2016]
x_test=df.drop(['total'],axis=1)[df['date'].dt.year>=2017]

y_train=df['total'][df['date'].dt.year<=2016].values
y_test=df['total'][df['date'].dt.year>=2017].values

x_train.drop('date',axis=1,inplace=True)
x_test.drop('date',axis=1,inplace=True)


model=LinearRegression()
model.fit(x_train,y_train)

st.markdown("<h1 style='text-align: center; color: red;'>IPL 2021 Score Predictor</h1>", unsafe_allow_html=True)

bat_team = st.selectbox('Select Batting Team',('Chennai Super Kings','Delhi Daredevils','Kings XI Punjab','Kolkata Knight Riders','Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore','Sunrisers Hyderabad'))
bowl_team = st.selectbox('Select Bowling Team',('Chennai Super Kings','Delhi Daredevils','Kings XI Punjab','Kolkata Knight Riders','Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore','Sunrisers Hyderabad'))

if bowl_team==bat_team:
    st.error('Bowling and Batting teams should be different')

for teams in bat_team_dict:
    if teams==bat_team:
        bat_team=bat_team_dict[teams]

for teams in bowl_team_dict:
    if teams==bowl_team:
        bowl_team=bowl_team_dict[teams]

overs=st.number_input('Enter The Overs',min_value=5.1,max_value=19.5,value=5.1,step=0.1)
if overs-math.floor(overs)>0.5:
    st.error('Please enter valid over input as one over contains only 6 balls')
wickets=st.slider('Enter Wickets Fallen Till Now',0,9)
runs = st.number_input('Current Team Score',min_value=0,max_value=354,step=1,format='%i')
runs_last_5=st.number_input('Runs Scored in Last 5 Overs',min_value=0,max_value=runs,step=1,format='%i')
wickets_last_5=st.slider('Wickets Taken in Last 5 Overs',0,9)

if st.sidebar.button('Project Details'):
    st.sidebar.write('Predict the run score of an IPL match using Linear Regression')
    st.sidebar.write('Dataset Link - https://drive.google.com/file/d/112j15X0GDb-sbLmoyk7i7Q9QaXtfukEG/view?usp=sharing')
if st.sidebar.button('IPL Details'):
    st.sidebar.write('The Indian Premier League (IPL) is a professional Twenty20 cricket league, contested by eight teams based out of eight different Indian cities.[3] The league was founded by the Board of Control for Cricket in India (BCCI) in 2007. It is usually held between March and May of every year and has an exclusive window in the ICC Future Tours Programme.')
    st.sidebar.write('Source Link - https://en.wikipedia.org/wiki/Indian_Premier_League')
if st.button("Predict Score"):
    output=int(model.predict([[bat_team,bowl_team,overs,wickets,runs,runs_last_5,wickets_last_5]])[0])
    st.write("Score Would Most Probably Range From "+str(output-5)+" To "+str(output+5))
