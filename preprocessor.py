import re
import streamlit as st
import pandas as pd
import text_transform
import helper


@st.cache_data
def preprocess_text(data):

    pattern=r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:am|pm)\s-\s(.*?):\s'
    name=re.findall(pattern, data)

    pattern=r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:am|pm)\s-\s.*?:\s(.*)'
    msg=re.findall(pattern, data)

    pattern=r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:am|pm))\s-\s.*?:\s'
    date=re.findall(pattern, data)

    df=pd.DataFrame({'msg_date':date, 'Name':name, 'msg':msg})
    df['msg_date']=pd.to_datetime(df['msg_date'], format='%d/%m/%Y, %I:%M %p')

    df['Date'] = df['msg_date'].dt.date
    df['year']=df['msg_date'].dt.year
    df['month']=df['msg_date'].dt.strftime("%b")
    df['day_name']=df.msg_date.dt.strftime("%a")
    df['hour']=df['msg_date'].dt.hour
    # df['day']=df['msg_date'].dt.day
    # df['minute']=df['msg_date'].dt.minute
    df['Time']=df['msg_date'].dt.strftime("%H:%M")
    df['period']=df['hour'].apply(lambda x: f"{x}-{(x+1)%24}")

    df["transformed_msg"] = df["msg"].apply(text_transform.transformed)

    return df