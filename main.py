#Environment variables
import requests
import os


#LangChain
from langchain.llms import OpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain


#Whisper
import whisper
from whisper.utils import write_vtt


#Streamlit
import streamlit as st
from streamlit_extras.buy_me_a_coffee import button
from annotated_text import annotated_text, annotation


#Set API keys
if "openai_api_key" in st.secrets:
    OPENAI_API_KEY = st.secrets["openai_api_key"]
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Functions
# Function to load Youtube video data and create transcripts
def get_yt_transcripts(url):
    st.write("Getting YouTube Videos...")
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    documents = loader.load()
    st.write("Creating Transcript...")
    return [(doc.page_content, doc.metadata) for doc in documents]

def summarize_transcripts(transcripts, chunk_size=2000, chunk_overlap=200):
    """
    Summarize the given transcripts.
    
    Parameters:
        transcripts (list): List of transcripts.
        chunk_size (int, optional): Size of the chunks the text should be split into.
        chunk_overlap (int, optional): Number of characters of overlap between chunks.
        
    Returns:
        list: Summarized text.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    summaries = []
    
    for transcription, metadata in transcripts:
        try:
            summaries.extend(text_splitter.split_documents(transcription))
        except Exception as e:
            print(f"Error processing transcript: {e}")

    return summaries


# Function to transcribe via Whisper
def transcribe_with_whisper(url):
    model = whisper.load_model("small")
    result = model.transcribe(url)
    return result


#Start Streamlit
# Set the Streamlit page configuration
st.set_page_config(page_title="Your Handy dandy transcription app", page_icon="🐱", layout="wide")

# Start Top Information
st.title("Your handy dandy transcription app 🐱")
st.markdown("### A mini-app to easily generate transcripts for YT videos and more.")

# End Top Information

# Output type selection by the user
input_type = st.radio(
    "Input Type:",
    ('Youtube URL', 'Other URLs', 'Local file'), horizontal=True)

output_type = st.radio(
    "Choose your output:",
    ('Transcript only', 'Transcript with summary'), horizontal=True)

if input_type == 'Youtube URL':
    youtube_videos = st.text_input(label="Youtube URLs (Separate multiple URLs with commas)", placeholder="Ex: https://www.youtube.com/watch?v=dQw4w9WgXcQ, https://www.youtube.com/watch?v=anothervideo", key="yt_videos")

    # Add a button
    button_pressed = st.button("Get Transcript/s")

    # Execute this block if the button is pressed
    if button_pressed:
        if youtube_videos:
                video_urls = youtube_videos.split(",")
                for url in video_urls:
                    url = url.strip()
                    transcripts = get_yt_transcripts(url)
                    for i, (transcription, metadata) in enumerate(transcripts):
                        st.text(f"Transcription for '{metadata['title']}'")
                        st.image(metadata['thumbnail_url'], caption=metadata['title'])
                        st.text_area(label=f"Transcription for '{metadata['title']}'", value=transcription, height=200, max_chars=None)
        
                        # Summarize the transcript and display the summaries if output_type is 'Transcript with summary'
                        if output_type == 'Transcript with summary':
                            summaries = summarize_transcripts([(transcription, metadata)])
                            for summary in summaries:
                                st.text_area(label=f"Summary for '{metadata['title']}'", value=summary, height=100, max_chars=None)
                        st.write("---")  # Add a separator between transcripts

        else:
            st.warning("Please enter Youtube URLs before pressing the button.")



elif input_type == 'Other URLs':
    annotated_text(annotation("🚨🚧 This section is under maintenance 🚧🚨", color="#8ef", border="1px dashed red"))
    other_urls = st.text_input(label="Other URLs (Separate multiple URLs with commas)", placeholder="Ex: ", key="other_urls")
    #model = whisper.load_model("small")

    # Add a button
    button_pressed = st.button("Get Transcript/s")

    # Execute this block if the button is pressed
    if button_pressed:
        if other_urls:
            video_urls = other_urls.split(",")
            for url in video_urls:
                    url = url.strip()
                    transcripts = get_yt_transcripts(url)
                    for i, (transcription, metadata) in enumerate(transcripts):
                        st.text(f"Transcription for '{metadata['title']}'")
                        st.image(metadata['thumbnail_url'], caption=metadata['title'])
                        st.text_area(label=f"Transcription for '{metadata['title']}'", value=transcription, height=200, max_chars=None)
        
                        # Summarize the transcript and display the summaries if output_type is 'Transcript with summary'
                        if output_type == 'Transcript with summary':
                            summaries = summarize_transcripts([(transcription, metadata)])
                            for summary in summaries:
                                st.text_area(label=f"Summary for '{metadata['title']}'", value=summary, height=100, max_chars=None)
                        st.write("---")  # Add a separator between transcripts

        else:
            st.warning("Please enter Youtube URLs before pressing the button.")



elif input_type == 'Local file':
    annotated_text(annotation("🚨🚧 This section is under maintenance 🚧🚨", color="#8ef", border="1px dashed red"))
    uploaded_files = st.file_uploader("Choose a file")

    #model = whisper.load_model("small")