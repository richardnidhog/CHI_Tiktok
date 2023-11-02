import os
import openai
import streamlit as st
import pandas as pd

st.title("Automatic TikTok Misinformation Analysis")
video_files = st.file_uploader('Upload Your Video File', type=['wav', 'mp3', 'mp4'], accept_multiple_files=True)

openai.api_key = "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" #insert API key

def transcribe(video_file):
    result = openai.Audio.transcribe("whisper-1", video_file)
    text = result["text"]
    transcript = '"{}"'.format(text)
    return transcript

def analyze(transcript):
    # Define the system message
    system_msg = 'You are a helpful assistant who can understand when a statement contains misinformation.'

    # Define the user message
    user_msg = 'Determine if the statement above contains any misinformation. In the part one, the results are first displayed in the first line, \'May contain misinformation\', \'Cannot be recognized at this time\' or \'No misinformation detected\'. In the part two, an operation is performed to extract keywords from the input text, providing up to six keywords, sorted in order of criticality. In the part three, briefly summarize the reasons for determining whether it contains misinformation. Three reasons of no more than 50 words each are required. Add line breaks between each section. Regardless of the state of the text given, it must be answered in the format given above:' + transcript

    # Create a dataset using GPT
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=[{"role": "system", "content": system_msg},
                                            {"role": "user", "content": user_msg}])
    
    content = response['choices'][0]['message']['content']
    gpt_response = '{}'.format(content)
    return gpt_response # find a way to extract GPT text and store in variable, follow same format (transcript = '"{}"'.format(text))

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
            st.sidebar.success('Transcribing' + " " + video_file.name)
            transcript = transcribe(video_file)
            analysis = analyze(transcript)
            st.markdown(video_file.name + ": " + transcript)
            st.markdown(video_file.name + ": " + analysis)
            
            d = {"transcript": transcript, "analysis": analysis}
            data.append(d)
            #print('"{}"'.format(transcript))
        else:
            st.sidebar.error("Please upload a video file")
    df = pd.DataFrame(data)
    csv = convert_df(df)
    st.download_button(
        label="Download transcipts as CSV",
        data=csv,
        file_name='transcripts.csv',
        mime='transcripts/csv',
    )