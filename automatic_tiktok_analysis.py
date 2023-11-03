import os
import openai
import streamlit as st
import requests
import pandas as pd
import datetime

st.title("Automatic Misinformation Analysis")



video_files = st.file_uploader('Upload Your Video File', type=['wav', 'mp3', 'mp4'], accept_multiple_files=True)

openai.api_key = "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" 

def transcribe(video_file):
    result = openai.Audio.transcribe("whisper-1", video_file)
    text = result["text"]
    transcript = '"{}"'.format(text)
    return transcript

def analyze(transcript):
    system_msg1 = "Your task is to determine if the following statement contains any misinformation. Begin by stating \'May contain misinformation\', \'Cannot be recognized\' or \'No misinformation detected\'."

    system_msg2 = "Your next task is to extract up to six keywords from the statement, sorted in order of criticality. Follow the format \"Keywords: ...\""

    system_msg3 = "Lastly, you must briefly summarize the reasons for determining whether the statement contains misinformation. Provide three or less reasons of no more than 50 words each."

    chat_sequence = [
        {"role": "system", "content": "You are a helpful assistant. You need to complete the requirements based on the following paragraph." + transcript},
        {"role": "user", "content": system_msg1},
        {"role": "user", "content": system_msg2},
        {"role": "user", "content": system_msg3}
    ]   

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_sequence
    )

    content = response['choices'][0]['message']['content']
    gpt_response = '{}'.format(content)
    return gpt_response

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

if st.button('Transcribe and Analyze Videos'):
    transcipt_status = False
    data = []
    for video_file in video_files:
        print(video_file.name)
        if video_file is not None:
            st.sidebar.success('Analyzing' + " " + video_file.name)
            transcript = transcribe(video_file)
            analysis = analyze(transcript)
            st.markdown(video_file.name + ": " + transcript)
            st.markdown(video_file.name + ": " + analysis)

            # Split the analysis information
            analysis = analysis.split('\n')
            misinformation_status = analysis[0]
            keywords = analysis[2]
            #reasons = ' '.join(analysis[4:])

            d = {
                "video_file": video_file.name,
                "transcript": transcript, 
                "misinformation_status": misinformation_status, 
                "keywords": keywords, 
                #"reasons": reasons
            }
            data.append(d)
            #print('"{}"'.format(transcript))        
        else:
            st.sidebar.error("Please upload a video file")
    df = pd.DataFrame(data)
    csv = convert_df(df)
    date_today = datetime.date.today().strftime('%Y_%m_%d')
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name = f'results_{date_today}.csv',
        mime='text/plain',
    )