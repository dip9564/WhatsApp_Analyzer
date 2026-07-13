import zipfile
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessor , helper, gemini
import streamlit.components.v1 as components
from google.genai.errors import ClientError

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fimg.freepik.com%2Fphotos-gratuite%2Fabstrait-numerique-grille-fond-noir_53876-97647.jpg%3Fsemt%3Dais_hybrid%26w%3D740&f=1&nofb=1&ipt=3a551da015306bbddd270ba2ae800e19c0cd7a495531591dc2fac4df03aeca4c");
    background-size: cover;
}

[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}
</style> """
st.markdown(page_bg_img,unsafe_allow_html=True)


Home,ChatScope,Sentiment,Compare,Summarize=st.tabs(["Home", "ChatScope","Sentiment","Compare Mode","Summarize"])

with Home:    
    st.title("WhatsApp Chat Analyzer 📊")
    uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp Chat(.txt or .zip)", type=["txt", "zip"])

    if uploaded_file is None:
        st.markdown(" \n\n Analyze WhatsApp chats with interactive visualizations, AI-powered summaries, sentiment analysis, and conversation insights.")
        st.subheader("How to Use :")

        st.markdown(""" 
        1. Export a WhatsApp chat from your WhatsApp Export Chat(Without Media recommended).
        2. Upload the TXT or ZIP file using the file uploader on the left sidebar.
        3. Choose a participant or select Overall.
        4. Click 'Show Analysis'

        ### ✨ Features

        - 📊 Message Statistics
        - 📈 Activity Timeline
        - 🔥 Weekly Heatmap
        - ☁️ Word Cloud
        - 😊 Emoji Analysis
        - 🧠 Sentiment Analysis
        - 🤝 Interaction Network
        - ⚖️ Member Comparison
        - 🤖 AI Chat Summary
        - 📂 Message Explorer
        """)
        left, center, right = st.columns([1, 2.5, 1])
        center.markdown("Built with Python • Streamlit • Scikit-learn • Gemini AI")

        ChatScope.warning("""No Response present, Please upload your file first !""")
        Sentiment.warning("""No Sentiment present, Please upload your file first !""")
        Compare.warning(""" No data is present for comparing, Please upload your file first !""")
        Summarize.warning(""" No data is present for summarizing, Please upload your file first !""")

    # Process the uploaded file
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                try:
                    txt_file = next(
                        f for f in zip_ref.namelist()
                        if f.endswith(".txt")
                    )
                except StopIteration:
                    st.error("No .txt file found in the ZIP archive.")
                    st.stop()

                with zip_ref.open(txt_file) as f:
                    data = f.read().decode("utf-8")
        else:
            data = uploaded_file.getvalue().decode("utf-8")


        # Preprocess the chat data
        df = preprocessor.preprocess_text(data)
        df = preprocessor.add_sentiment(df)

        if df.empty:
            st.error("This does not appear to be a valid WhatsApp chat export. Please upload a valid file.")
            st.stop()

        st.sidebar.header("Select Analysis Type")
        user_list = df.Name.sort_values().unique()

        user = st.sidebar.selectbox("Choose specific person", ["Overall"] + user_list.tolist())

        # Initialize session state for analysis visibility
        if "analysis_shown" not in st.session_state:
            st.session_state.analysis_shown = False

        if st.sidebar.button("Show Analysis"):
            st.session_state.analysis_shown = True

        if st.session_state.analysis_shown:

            if user == "Overall":
                st.header("📊 Overview")
            else:
                st.header(f"Statistics of : {user}")

            total_messages, total_words, total_media_files, total_links = helper.total_messages(df,user)
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("💬 Total messages :",f"{ total_messages}")
            col2.metric("📝 Total words :",f"{ total_words}")
            col3.metric("🖼️ Total media files :",f"{ total_media_files}")
            col4.metric("🔗 Total Links :",f"{ total_links}")

            MAX_MESSAGES = 100000
            if len(df) > MAX_MESSAGES:
                df = df.tail(MAX_MESSAGES)

            # group analysis
            col1, col2 = st.columns(2)
            if user == "Overall":
                col1.success(f"Most active person: { df.Name.value_counts().index[0] } ")

                if df.Name.value_counts().index[-1] =="Meta AI":
                    col2.warning(f"Most silent person: { df.Name.value_counts().index[-2] }")
                else:
                    col2.warning(f"Most silent person: { df.Name.value_counts().index[-1] }")



                user_df=helper.user_persentage(df)
                st.subheader(f"👥 Participant Distribution ({user_df.shape[0]} Members)")
                col1, col2 = st.columns([1.3,2])

                if user_df.shape[0] > 5:
                    col1.dataframe(user_df,hide_index=True,height=250)
                else:
                    col1.dataframe(user_df,hide_index=True)

                plt.style.use("dark_background")
                fig,ax=plt.subplots(figsize=(4,4))
                user_ls=user_df['Percentage'].head().tolist()
                label=user_df['Name'].head().tolist()

                ax.pie(user_ls, labels=label, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
                ax.axis('equal')
                col2.pyplot(fig)
                plt.close(fig)


            # monthly timeline
            st.subheader("📈 Monthly Activity Trend :")
            timeline=helper.monthly_timeline(df,user)

            plt.style.use("dark_background")
            fig,ax=plt.subplots(figsize=(10,6))

            ax.plot(timeline['time'], timeline['msg'],color="#20b540")
            plt.xticks(rotation='vertical')
            plt.ylabel("No. of massages")
            st.pyplot(fig)
            plt.close(fig)

            # daily_activity
            st.header(' 📅 Activity Overview')
            col2,col1=st.columns(2)

            col1.subheader("📆 Messages by Weekday:")
            month_df,daily_df,hour_df= helper.weekly_activity(df,user)
            fig,ax=plt.subplots()

            ax.bar(daily_df['day_name'], daily_df['count'],color="#2261e9")
            plt.ylabel("No. of massages")
            col1.pyplot(fig)
            plt.close(fig)

            # monthly_activity
            col2.subheader("🗓️ Messages by Month:")
            fig,ax=plt.subplots()

            ax.bar(month_df['month'], month_df['count'],color="#2261e9")
            plt.ylabel("No. of massages")
            col2.pyplot(fig)
            plt.close(fig)

            # hourly_activity
            st.subheader("⏰ Messages by Daily:")
            fig,ax=plt.subplots(figsize=(16,6))

            ax.bar(hour_df['period'], hour_df['count'],color="#2261e9")
            plt.xticks(rotation='vertical')
            plt.ylabel("No. of massages")
            st.pyplot(fig)
            plt.close(fig)

            # Activity Heatmap
            st.subheader("Weekly Activity Heatmap:")
            fig, ax = plt.subplots(figsize=(16, 6))

            heatmap_df=helper.heatmap(df,user)
            sns.heatmap(heatmap_df,annot=True,fmt=".0f", cmap="YlGnBu")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
            plt.close(fig)

            st.header(' 💬 Content Analysis')
            # word_cloud
            col1, col2 = st.columns([3,1])
            col1.markdown("#### ☁️ Most Frequent Words :")
            df_wc=helper.word_cloud(df,user)
            fig, ax=plt.subplots()
            ax.imshow(df_wc)
            ax.axis("off")
            col1.pyplot(fig)
            plt.close(fig)

            # emoji
            col2.markdown("##### 😊 Top Emojis:")
            emoji_df=helper.emoji_count(df,user)

            col2.dataframe(emoji_df)



#-------------------------------------------------------------------------------------------------#
            with ChatScope:
                st.header("Response Analysis")
                response_df,avg_restime,median_restime= helper.response_time(df,user)

                col1, col2 = st.columns([1.5,2])
                if user == "Overall":
                    col1.metric(
                        label= "⏱️ Average Response Time : ",
                        value=f"{avg_restime:.2f} sec"
                    )
                    col1.metric(
                        label= "⏱️ Median Response Time : ",
                        value=f"{median_restime:.2f} sec"
                    )
    
                    if response_df.Name.iloc[0] == 'Meta AI':
                        col2.success(f"⚡ Fastest Responder: { response_df.Name.iloc[1] } ")
                    else:
                        col2.success(f"⚡ Fastest Responder: { response_df.Name.iloc[0] } ")
                    col2.warning(f"🐢 Slowest Responder: { response_df.Name.iloc[-1] }")

                    st.subheader(f"⏱️ Response Time Ranking ({response_df.shape[0]-1} users) :")
                    response_df=response_df.sort_values('Name').reset_index(drop=True).head(25)
                    fig,ax=plt.subplots(figsize=(10,4))

                    ax.bar(response_df['Name'], response_df['Response_time'],color="#13c2d6")
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    col1.metric(
                        label=f"{user} 's Average Response Time :",
                        value=f"{avg_restime:.2f} sec"
                    )
                    col2.metric(
                        label=f"{user}'s Median Response Time :",
                        value=f"{median_restime:.2f} sec"
                    )
#-------------------------------------------------------------------------------------------------#
                try:
                    st.subheader("🌒 Night Owl Analysis : ")
                    night_df,night_owl,night_msg=helper.night_conversion(df,user)

                    if user == "Overall":
                        st.markdown(f"""
                            ##### Night Owl 🦉 :
                            **{night_owl}** \n
                            **{night_msg} Messages sent between 10 PM and 6 AM.**
                            """)

                        st.markdown("### Messages at Night:")
                        fig,ax=plt.subplots(figsize=(10,4))

                        ax.bar(night_df['Name'], night_df['count'],color="#13c2d6")
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.metric("Total message in Night: ",night_msg)
                except Exception as e:
                    st.error(f"Night Messaging Analysis not available ")

#-------------------------------------------------------------------------------------------------#
    
                net,most_talker = helper.interaction_graph(df,user)

                if user == "Overall":
                    st.markdown(f"""
                        ### 💬 Best Interactive Partners :
                        **{most_talker['source']} ⇔ {most_talker['target']}** \n
                        **{most_talker['count']} Interactions**
                    """)
                else: 
                    st.markdown(f"""
                        #### 💬 Talk Most with :
                        **{most_talker['source']} ⎯➣ {most_talker['target']}** \n
                        **{most_talker['count']}** Interactions
                    """)

                st.markdown(" #### 🕸️ Interaction Network : ")

                net.save_graph("interaction_graph.html")
                with open("interaction_graph.html", "r", encoding="utf-8") as f:
                    html = f.read()

                html = html.replace(
                        "<body>","<body style='background-color: transparent;'>"
                        """ <script>
                            network.once("stabilizationIterationsDone", function () {
                                network.stopSimulation();
                                network.fit({
                                    animation: {
                                        duration: 500,
                                        easingFunction: "easeInOutQuad"
                                    }
                                });
                            });
                            </script>
                        </body>"""
                    )

                components.html(html, height=410,width=None,)


#-------------------------------------------------------------------------------------------------#
                conversation_df = helper.conversation_analysis(df)
                if user == "Overall":
                    st.subheader(" Conversation Initiators : ")

                    name=conversation_df.loc[conversation_df.conversation_started.idxmax(),'Name']
                    st.markdown(f"""
                        ##### 🗣️ Top Conversation Starter :
                        **{ name }**
                    """)

                    fig,ax=plt.subplots(figsize=(10,6))
                    user_ls=conversation_df['conversation_started'].head().tolist()
                    label=conversation_df['Name'].head().tolist()

                    ax.pie(user_ls, labels=label, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
                    ax.axis('equal')
                    st.pyplot(fig)
                    plt.close(fig)
#-------------------------------------------------------------------------------------------------#

            with Summarize:
                #  Message Explorer
                min_date = df['Date'].min()
                max_date = df['Date'].max()

                st.header(" AI Summary")

                with st.form("Date_form"):
                    col1, col2 = st.columns(2)
                    Date_range = col1.date_input('Select the conversation period to summarize :',value=(max_date - pd.Timedelta(days=1), max_date),
                           min_value=min_date,
                           max_value=max_date)
                    

                    apply = col1.form_submit_button("✨ Generate Summary")

                    if apply:
                        try:
                            with st.spinner("Generating summary..."):   
                                summary = gemini.summarize_chat(df, user, Date_range)
                                st.subheader("Summary of the WhatsApp Conversation:")
                                st.write(summary)
                        except ClientError as e:
                            st.error(f"Gemini quota exceeded. Please try again later.")
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")
                    else :
                        st.info("Selected specific person for his chat summary")


                st.sidebar.info(f"Chat data available from: \n\n {min_date} to {max_date}")

                with st.expander("📂 Message Explorer:"):
                    with st.form("date_form"):
                        date_range = st.date_input('Choose Date range:',value=(max_date - pd.Timedelta(days=1), max_date),
                                                   min_value=min_date,
                                                   max_value=max_date)
                        show = st.form_submit_button("show messages")
                        if show:
                            masseges= helper.date_range_selector(df,user,date_range)
                            if masseges.shape[0] > 7:
                                st.dataframe(masseges,hide_index=True,height=310)
                            else:
                                st.dataframe(masseges,hide_index=True)
#-------------------------------------------------------------------------------------------------#

            with Sentiment:
                st.header("Sentiment Insights :")

                if uploaded_file is not None:
                    col1,col2=st.columns(2)

                    user_sentiment,positive,negetive,negative_msg,positive_msg=helper.sentiment_analysis(df,user)    
                

                    col1.metric("😊 Positive Messages: ",f"{round(positive,2)}%")
                    col2.metric("😔 Negative Messages : ",f"{round(negetive,2)}%")

                    if user == "Overall":
                        Negative_user = user_sentiment.Name.iloc[0]
                        Positive_user = user_sentiment.Name.iloc[-1]

                        col1.markdown(""" ### Most Positive participant :""")
                        col1.success(f"{Positive_user}")

                        col2.markdown(""" ### Most Negative participant :""")
                        col2.error(f"{Negative_user}")

                        fig, ax = plt.subplots(figsize=(10, 5))
                        bars = ax.barh(
                            user_sentiment["Name"],
                            user_sentiment["prositivity"],
                            color="#3B82F6",
                            height=0.65
                        )
                        ax.invert_yaxis()
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(
                                width + 0.5,
                                bar.get_y() + bar.get_height()/2,
                                f"{width:.1f}%",
                                va="center",
                                fontsize=10,
                                color="white"
                            )
                        ax.set_xlabel("Positive Sentiment (%)", fontsize=12)
                        ax.set_xlim(0, 100)
                        ax.grid(axis="x", linestyle="--", alpha=0.3)
                        ax.set_facecolor("#000000")
                        fig.patch.set_facecolor("#000000")

                        # Remove unnecessary borders
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    col1, col2 = st.columns([2,1])
                    with col1.form("sentiment_form"):
                        sentiment_count = st.number_input("Top Messages to Display:", min_value=1, max_value=25, value=5)
                        show_sentiment = st.form_submit_button("Show Sentiment Messages")

                    col1, col2 = st.columns([1,1])
                    top_negative = negative_msg.nlargest(sentiment_count, "proba")[["Name", "msg"]]
                    top_positive = positive_msg.nsmallest(sentiment_count, "proba")[["Name", "msg"]]
                    col1.subheader(f"Top {sentiment_count} Positive Messages")
                    col1.dataframe(top_positive,hide_index=True)
                    col2.subheader(f"Top {sentiment_count} Negative Messages")
                    col2.dataframe(top_negative,hide_index=True)


#-------------------------------------------------------------------------------------------------#
            with Compare:
                st.subheader("Compare between Two Member: ")
                with st.form("compare_form"):
                    col1,col2,col3=st.columns([2,1,2])
                    col2.markdown("## ⚔️VS⚔️ ")
                    person1 =col1.selectbox("Select first person :",user_list)
                    person2 = col3.selectbox("Select second person :",user_list)

                    compare = st.form_submit_button("show compare")

                if compare: 

                    col1, col2, col3, col4 = st.columns(4)
                    total_messages1, total_words1, total_media_files1, total_links1 = helper.total_messages(df,person1)
                    total_messages2, total_words2, total_media_files2, total_links2 = helper.total_messages(df,person2)

                    col1.metric("💬 messages:",total_messages1)
                    col2.metric("📝 words:",total_words1)
                    col1.metric("📷 media files:",total_media_files1)
                    col2.metric("🔗 Links:",total_links1)

                    col3.metric("💬 messages:",total_messages2)
                    col4.metric("📝 words:",total_words2)
                    col3.metric("📷 media files:",total_media_files2)
                    col4.metric("🔗 Links:",total_links2)

                    ## responce time
                    st.subheader("⏱️ Responce Time Analysis :")
                    col1, col2, col3, col4 = st.columns(4)
                    _, avg_restime1,med_restime1= helper.response_time(df,person1)
                    _, avg_restime2,med_restime2= helper.response_time(df,person2)

                    col1.metric("Average Response Time: ", f"{avg_restime1:.2f} sec")
                    col2.metric("Median Response Time: ", f"{med_restime1:.2f} sec")
                    col3.metric("Average Response Time: ", f"{avg_restime2:.2f} sec")
                    col4.metric("Median Response Time: ", f"{med_restime2:.2f} sec")


                    # interaction
                    col1,col2=st.columns(2)
                    _,most_talker1 = helper.interaction_graph(df,person1)
                    _,most_talker2 = helper.interaction_graph(df,person2)

                    col1.markdown(f"""
                        #### 💬 Talk Most with :
                        **{most_talker1['source']} ⎯➣ {most_talker1['target']}** \n
                        **{most_talker1['count']} Interactions**
                    """)
                    col2.markdown(f"""
                        #### 💬 Talk Most with :
                        **{most_talker2['source']} ⎯➣ {most_talker2['target']}** \n
                        **{most_talker2['count']} Interactions**
                    """)

                    st.subheader("🌒 Night Messageing : ")
                    try:
                        col1,col2=st.columns(2)
                        _,night_owl1,night_msg1=helper.night_conversion(df,person1)
                        _,night_owl2,night_msg2=helper.night_conversion(df,person2)

                        col1.metric("Night Messages: ",night_msg1)
                        col2.metric(" Night Messages: ",night_msg2)
                    except Exception as e:
                        st.error(f" Night Messaging not available for {person1} or {person2} ")

                    # sentiment
                    st.subheader("Sentiment Analysis :")
                    col1, col2, col3, col4 = st.columns(4)
                    try :

                        positive1=user_sentiment.loc[user_sentiment['Name'] == person1,'prositivity'].iloc[0]
                        negetive1=user_sentiment.loc[user_sentiment['Name']== person1,'negetivity'].iloc[0]

                        positive2=user_sentiment.loc[user_sentiment['Name'] == person2,'prositivity'].iloc[0]
                        negetive2=user_sentiment.loc[user_sentiment['Name']== person2,'negetivity'].iloc[0]

                        col1.metric("Positivity Rate : ",f"{round(positive1,2)}%")
                        col1.metric("Negativity Rate : ",f"{round(negetive1,2)}%")

                        col3.metric("Positivity Rate : ",f"{round(positive2,2)}%")
                        col3.metric("Negativity Rate : ",f"{round(negetive2,2)}%")
                    except:
                        col1.warning("Insufficient messages for sentiment analysis.")

                    emoji_df1=helper.emoji_count(df,person1)
                    emoji_df2=helper.emoji_count(df,person2)

                    col2.dataframe(emoji_df1, height=200)
                    col4.dataframe(emoji_df2, height=200)

                    # conversation start
                    try:
                        col1, col2= st.columns(2)

                        conversation_time1=conversation_df[conversation_df["Name"]==person1]['conversation_started'].values[0]
                        conversation_time2=conversation_df[conversation_df["Name"]==person2]['conversation_started'].values[0]
                        user_ls=[conversation_time1,conversation_time2]

                        name=[person1,person2]
                        conversation_per1= conversation_time1 *100 // (conversation_time1+conversation_time2)

                        col1.markdown(f""" ### 💬 Conversation Starter :""")
                        if conversation_time1 > conversation_time2:
                            col1.success(f"""**{ person1 }**""")
                            col1.markdown(f"""**Started {round(conversation_per1,2)}% of conversations**""")
                        else:
                            col1.success(f"""**{ person2 }**""")
                            col1.markdown(f"""**Started {round(100 - conversation_per1,2)}% of conversations**""")
                            
                        fig,ax=plt.subplots(figsize=(4,4))
                        ax.pie(user_ls, labels=name, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
                        ax.axis('equal')
                        col2.pyplot(fig)
                    except Exception :
                        st.error(f"Conversation not available for {person1} or {person2}")

                    ## timeline
                    st.subheader("Monthly activity compare :")
                    timeline1,daily1,hourly1=helper.weekly_activity(df,person1)
                    timeline2,daily2,hourly2=helper.weekly_activity(df,person2)

                    x = np.arange(len(timeline1['month']))
                    width=0.35
                    fig,ax=plt.subplots(figsize=(10,6))

                    ax.bar(x-width/2, timeline1['count'],width=width,color="#4cc232",label=person1)
                    ax.bar(x+width/2, timeline2['count'],width=width,color="#1868de",label=person2)

                    plt.xticks(ticks=x, labels=timeline1['month'])
                    plt.xticks(rotation='vertical')
                    plt.ylabel("No. of massages")
                    plt.legend()
                    st.pyplot(fig)
                    plt.close(fig)


                    st.subheader("Weekly activity compare :")
                    x = np.arange(len(daily1['day_name']))
                    width=0.35
                    fig,ax=plt.subplots(figsize=(10,6))

                    ax.bar(x-width/2, daily1['count'],width=width,color="#1ebce4",label=person1)
                    ax.bar(x+width/2, daily2['count'],width=width,color="#1868de",label=person2)

                    plt.xticks(ticks=x, labels=daily1['day_name'])
                    plt.xticks(rotation='vertical')
                    plt.ylabel("No. of massages")
                    plt.legend()
                    st.pyplot(fig)
                    plt.close(fig)


                    st.subheader("Daily activity compare :")
                    x = np.arange(len(hourly1['period']))
                    width=0.35
                    fig,ax=plt.subplots(figsize=(10,6))

                    ax.bar(x-width/2, hourly1['count'],width=width,color="#BA723F",label=person1)
                    ax.bar(x+width/2, hourly2['count'],width=width,color="#1868de",label=person2)

                    plt.xticks(ticks=x, labels=hourly1['period'])
                    plt.xticks(rotation='vertical')
                    plt.ylabel("No. of massages")
                    plt.legend()
                    st.pyplot(fig)
                    plt.close(fig)




