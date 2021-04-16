import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import streamlit as st

df=pd.read_csv('ipl.csv')

df=df[df['overs']>=5.0]

df=df.drop(labels=['mid','batsman','bowler','striker','non-striker'],axis=1)

consistent_team=['Kolkata Knight Riders','Chennai Super Kings','Rajasthan Royals','Mumbai Indians',
                  'Kings XI Punjab','Royal Challengers Bangalore','Delhi Daredevils',
                  'Sunrisers Hyderabad']
## keeping only teams that exists now

df=df[df['bat_team'].isin(consistent_team) & df['bowl_team'].isin(consistent_team)]
## Converting datetime string into datetime object

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

st.markdown("<h1 style='text-align: center; color: red;'><b>IPL 2021 Match Prediction</b></h>",unsafe_allow_html=True)
bat_team=st.selectbox("Batting Team",('Kolkata Knight Riders','Chennai Super Kings','Rajasthan Royals','Mumbai Indians',
                  'Kings XI Punjab','Royal Challengers Bangalore','Delhi Daredevils',
                  'Sunrisers Hyderabad'))
bowl_team=st.selectbox("Bowling Team",('Kolkata Knight Riders','Chennai Super Kings','Rajasthan Royals','Mumbai Indians',
                  'Kings XI Punjab','Royal Challengers Bangalore','Delhi Daredevils',
                  'Sunrisers Hyderabad'))

for teams in bat_team_dict:
    if teams==bat_team:
        bat_team=bat_team_dict[teams]

for teams in bowl_team_dict:
    if teams==bowl_team:
        bowl_team=bowl_team_dict[teams]

overs=st.number_input("Overs")
wickets=st.number_input('Wickets Lost')
runs=st.number_input('Current Team Score')
rl5=st.number_input("Runs last 5 overs")
wl5=st.number_input("Wickets last 5 overs")


if st.button("Predict"):
            output=int(model.predict([[bat_team,bowl_team,overs,wickets,runs,rl5,wl5]])[0])
            st.write("Score may range from "+str(output-10)+" to "+str(output))
