import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import emoji
import pickle
from pyvis.network import Network

@st.cache_resource
def total_messages(df,user):
    words = []
    link=0
    shape = 0

    if user != "Overall":
        df=df[df['Name']==user]

    for message in df.msg:
        words.extend(message.split())

    media_files = df[df.msg == "<Media omitted>"]

    for wrd in df.msg:
        if "https://" in wrd or "http://" in wrd:
            link+=1

    return df.shape[0], len(words), media_files.shape[0], link

@st.cache_data
def user_persentage(df):
    user_df=pd.DataFrame(df.Name.value_counts().reset_index())
    user_df['Percentage']=round((df.Name.value_counts()/df.shape[0])*100).values
    return user_df

@st.cache_data
def word_cloud(df,user):

    df = df[df.msg != "<Media omitted>"]
    if user != "Overall":
        df=df[df['Name']==user]

    wc=WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    return wc.generate(df['msg'].str.cat(sep=" "))

@st.cache_data
def emoji_count(df,user):
    if user != "Overall":
        df=df[df['Name']==user]

    emojis=[]
    for massege in df.msg:
        emojis.extend([c for c in massege if emoji.is_emoji(c)])
    if len(emojis) == 0:
        return pd.DataFrame(columns=['emoji', 'count'])
    emoji_df=pd.DataFrame(emojis)
    emoji_df.columns=['emoji']
    return emoji_df.value_counts()

@st.cache_data
def monthly_timeline(df,user):
    if user != "Overall":
        df=df[df['Name']==user]

    timeline=df.groupby(['year','month']).count()['msg'].reset_index()
    timeline['time']=timeline['month']+'-'+timeline['year'].astype(str)

    month_order=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    timeline['month']= pd.Categorical(timeline['month'], categories=month_order, ordered=True)

    timeline=timeline.sort_values(by=["year","month"])
    return timeline

@st.cache_data
def weekly_activity(df,user):
    if user != "Overall":
        df=df[df['Name']==user]

    daily_order = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    daily_df = (
        df.groupby('day_name').size().reindex(daily_order, fill_value=0).reset_index(name='count')
    )

    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_df = (
        df.groupby('month').size().reindex(month_order, fill_value=0).reset_index(name='count')
    )

    hour_order = [f"{i}-{(i+1)%24}" for i in range(24)]
    hour_df = (
        df.groupby('period').size().reindex(hour_order, fill_value=0).reset_index(name='count')
    )

    return month_df,daily_df, hour_df

@st.cache_data
def heatmap(df,user):
    if user != "Overall":
        df=df[df['Name']==user]

    
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df["day_name"] = pd.Categorical(df["day_name"],categories=days,ordered=True)

    periods = [f"{i}-{(i+1)%24}" for i in range(24)]
    df["period"] = pd.Categorical(df["period"],categories=periods,ordered=True)

    heatmap_df = df.pivot_table(
    index="day_name",
    columns="period",
    aggfunc="size"
    ).fillna(0)
    return heatmap_df

@st.cache_data
def date_range_selector(df,user,date_range):

    if user != "Overall":
        df=df[df['Name']==user]

    if len(date_range) == 2:
        min_date = date_range[0]
        max_date = date_range[1]
    else:
        min_date = max_date = date_range[0]

    filt =(df.Date >=min_date) & (df.Date <= max_date)
    messages=df.loc[filt,['Date', 'Time', 'msg']]

    messages.rename(columns={'msg':'Messages'},inplace=True)

    return messages

@st.cache_resource
def Sentiment_col(df):
    pipeline = pickle.load(open("whatsApp_pipeline.pkl", "rb"))
    proba = pipeline.predict(df.transformed_msg)

    return proba

@st.cache_resource
def Sentiment(df,user):

    pipeline = pickle.load(open("whatsApp_pipeline.pkl", "rb"))
    pred = pipeline.predict(df.transformed_msg)
    df['pred'] = pred
    user_sentiment = (df.groupby('Name')['pred']
      .value_counts(normalize=True)
      .mul(100)
      .unstack(fill_value=0)
      .reset_index()
    )
    user_sentiment.columns=['Name','negetivity','prositivity']

    if user == "Overall":
        positive=user_sentiment['prositivity'].mean()
        negetive=user_sentiment['negetivity'].mean()
    else:
        positive=user_sentiment.loc[user_sentiment['Name'] == user,'prositivity'].iloc[0]
        negetive=user_sentiment.loc[user_sentiment['Name']== user,'negetivity'].iloc[0]

    msg_count = df.groupby('Name').size()
    active_users = msg_count[msg_count >= 20].index
    user_sentiment = user_sentiment[user_sentiment['Name'].isin(active_users)]
    user_sentiment.sort_values(by='negetivity',ascending=False,inplace=True)

    return user_sentiment,positive,negetive

@st.cache_resource
def Sentiment_msg(df,user,sentiment_count):
    if user != "Overall":
        df=df[df['Name']==user]
    pipeline = pickle.load(open("whatsApp_pipeline.pkl", "rb"))
    proba = pipeline.predict_proba(df.transformed_msg)

    df['proba']=proba[ :, 0]


    top_negetive = df.nlargest(sentiment_count,'proba')[['Name','msg']]
    top_negetive.columns = ['Name', 'Message']

    top_positive = df.nsmallest(sentiment_count,'proba')[['Name','msg']]
    top_positive.columns = ['Name', 'Message']

    return top_negetive, top_positive

@st.cache_data
def response_time(df,user):
    df["prev_user"] = df["Name"].shift(1)
    df["prev_time"] = df["msg_date"].shift(1)

    responses_df = df[df["Name"] != df["prev_user"]].copy()
    responses_df["Response_time"] = (responses_df["msg_date"] - responses_df["prev_time"]).dt.total_seconds() / 60

    responses_df=responses_df[responses_df["Response_time"] <= 120]
    avg_restime= int(responses_df["Response_time"].mean())

    responses_time=responses_df.groupby("Name")["Response_time"].mean().reset_index()

    filt= (responses_time["Name"] == "Meta AI")
    responses_time.loc[filt, "Response_time"] = 0
    responses_time.sort_values(by='Response_time',inplace=True)
    responses_time['Response_time']=responses_time['Response_time'].apply(lambda x : round(x,0))

    median_df=responses_df.groupby("Name")["Response_time"].median().reset_index()
    median_time=median_df["Response_time"].mean()

    if user != "Overall":
        avg_restime = responses_time[responses_time["Name"] == user]["Response_time"].values[0]
        median_time= median_df[median_df["Name"] == user]["Response_time"].values[0]

    smallest_df = responses_time.nsmallest(25, "Response_time")

    return smallest_df, avg_restime, median_df, median_time

@st.cache_data
def conversation_analysis(df):
    df["prev_time"] = df["msg_date"].shift(1)
    df["gap"] = ( df["msg_date"] - df["prev_time"]).dt.total_seconds() / 60

    conversation_starters = df[df["gap"] > 120]
    conversation_starters = (conversation_starters.groupby("Name").size().reset_index(name="conversation_started"))
    conversation_starters.sort_values(by='conversation_started',ascending=False,inplace=True)

    conversation_starters['conversation_started']=conversation_starters['conversation_started'].apply(lambda x : x/446 *100)
    
    return conversation_starters

@st.cache_data
def bidirectional(interaction_df):
    # Create an unordered pair
    interaction_df["pair"] = interaction_df.apply(
        lambda x: tuple(sorted([x["source"], x["target"]])),
        axis=1
    )
    # Sum the counts for each pair
    bidirectional_df = (
        interaction_df
        .groupby("pair", as_index=False)["count"]
        .sum()
    )
    # Split the pair back into source and target
    bidirectional_df[["source", "target"]] = pd.DataFrame(
        bidirectional_df["pair"].tolist(),
        index=bidirectional_df.index
    )
    # Remove the temporary column
    bidirectional_df = bidirectional_df.drop(columns="pair")
    return bidirectional_df

@st.cache_data
def interaction_analysis(df,user):

    reply_df = df[df['Name'] != df['Name'].shift()].copy()
    reply_df['previous_user'] = (reply_df['Name'].shift())
    reply_df=reply_df[['Name','previous_user']]
    reply_df= reply_df.iloc[1:]

    interaction_df=reply_df.groupby(['previous_user', 'Name']).size().reset_index(name='count')
    interaction_df.columns = ['target', 'source', 'count']

    if user != "Overall": 
        filter=(interaction_df['source'] == user) 
        interaction_df = interaction_df[filter]

    else:
        interaction_df = bidirectional(interaction_df)
        filt= interaction_df['count'] > interaction_df['count'].mean()/10
        interaction_df = interaction_df[filt]

    return interaction_df

@st.cache_data
def interaction_graph(df,user):
    interaction_df =interaction_analysis(df,user)

    net = Network(height="400px", width="100%", directed=True,bgcolor="#0E1117",font_color="white")
    net.set_options("""
        var options = {
          "nodes": {
            "shape": "dot",
            "font": {
              "size": 10,
              "color": "white"
            }
          },
          "edges": {"smooth": true}
        }
        """)

    users = set(interaction_df["source"]).union(interaction_df["target"])
    msg_count = df["Name"].value_counts()
    for usr in users:
        net.add_node(
            usr,
            size=20,
            borderWidth=2,
            title=f"{msg_count.get(usr,0)} messages",
            color={        
            "background": "#4FC3F7",
            "border": "#1565C0",
            "highlight": {
                "background": "#57F6E6",
                "border": "#123CE4"
                }
            },
            font={"size": 10,"color": "white"}
        )

    if user == "Overall":
        for _, row in interaction_df.iterrows():
            net.add_edge(
                row["source"],
                row["target"],
                borderWidth=1,
                value=row["count"],
                title=f"{row['count']} replies",
                arrows="",
                color={"color": "#2636E4","highlight": "#5C33FF",}
            )
    else:
        for _, row in interaction_df.iterrows():
            net.add_edge(
                row["source"],
                row["target"],
                borderWidth=1,
                value=row["count"],
                title=f"{row['count']} replies",
                arrows="to",
                color={"color": "#2636E4","highlight": "#5C33FF",}
            )

    return net,interaction_df


def night_conversion(df,user):
    if user!="Overall":
        df=df[df["Name"]==user]

    night_df=df[(df["hour"] > 22) | (df["hour"] < 5)]
    nightOwl_df= night_df['Name'].value_counts().reset_index()

    night_owl=nightOwl_df.loc[[0],'Name'].values[0]
    night_msg=nightOwl_df.loc[[0],'count'].values[0]

    nightOwl_df=nightOwl_df.sort_values(by="Name").head(25)

    return nightOwl_df,night_owl,night_msg
