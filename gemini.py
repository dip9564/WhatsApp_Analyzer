from google import genai
import streamlit as st

# api_key = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key="api_key")

def summarize(chat_text):
    prompt = f"""
        You are an AI assistant that analyzes WhatsApp conversations.
        Analyze the conversation below and generate a clear, well-structured summary.

        Instructions:
        - Keep the summary concise but informative.
        - Do not repeat messages.
        - Ignore greetings, emojis, stickers, and small talk unless important.
        - Highlight only meaningful discussions.

        Return the result using these sections:

        ### 📌 Overview
        Provide a 2-3 sentence summary of the conversation.

        ### 💬 Main Discussion Topics
        List the major topics discussed.

        ### ✅ Important Decisions
        Mention any decisions that were made.
        If none, write "No important decisions."

        ### 📋 Action Items
        List any tasks, reminders, deadlines, or plans.
        If none, write "No action items."

        ### 😊 Overall Mood
        Describe the overall tone of the conversation
        (e.g., Positive, Neutral, Excited, Serious, Mixed).

        ### 👥 Key Participants
        Mention who contributed the most and what they mainly discussed.

        Conversation:
        {chat_text}
        """

    response = client.models.generate_content(
        model="models/gemini-3.1-flash-lite",
        contents=prompt
    )

    return response.text


def summarize_chat(df, user, date_range):
    if user!="Overall":
        df = df[df['Name'] == user]

    min_date = date_range[0]
    max_date = date_range[1]

    filt =(df.Date >=min_date) & (df.Date <= max_date)
    messages=df.loc[filt,['Date', 'msg']]

    MAX_MESSAGES = 2000

    messages = df["msg"].dropna().tolist()
    if len(messages) > MAX_MESSAGES:
        messages = messages[-MAX_MESSAGES:]  # last 2000 messages

    chat_text = "\n".join(messages)

    return summarize(chat_text)
