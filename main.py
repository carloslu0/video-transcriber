#Environment variables
import logging
import requests
import os
import os
from dotenv import load_dotenv  

# Configure logging
logging.basicConfig(level=logging.DEBUG)

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
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#Functions
def get_yt_transcripts(url):
    st.write("Getting YouTube Videos...")
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    documents = loader.load()
    st.write("Creating Transcript...")
    return [(doc.page_content, doc.metadata) for doc in documents]


# Function to split long transcripts into chunks
def summarize_transcripts(transcript):
    size = st.text_input(label="Enter chunk size:", placeholder="Ex: 2000")
    overlap = st.text_input(label="Enter chunk overlap:", placeholder="Ex:200")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    docs = text_splitter.create_documents([transcript])
    return docs


def inference(link):
  content = requests.get(link)
  podcast_url = re.findall("(?P<url>\;https?://[^\s]+)", content.text)[0].split(';')[1]
  print(podcast_url)
  

  download = requests.get(podcast_url)

  with open('podcast.mp3', 'wb') as f:
    f.write(download.content)

  result = model.transcribe('podcast.mp3')

  with open('sub.vtt', "w") as txt:
    write_vtt(result["segments"], file=txt)

  return (result['text'], 'sub.vtt')


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
    button_pressed = st.button("Get Transcripts")

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
                    st.write("---")  # Add a separator between transcripts
                    
                    if output_type == 'Transcript with summary':
                        summarized_transcripts = summarize_transcripts(transcription)
                        st.write("Summary:")
                        for doc in summarized_transcripts:
                            st.write(doc)
        else:
            st.warning("Please enter Youtube URLs before pressing the button.")



elif input_type == 'Other URLs':
    other_urls = st.text_input(label="Other URLs (Separate multiple URLs with commas)", placeholder="Ex: ", key="other_urls")
    model = whisper.load_model("small")



elif input_type == 'Local file':
    annotated_text(annotation("This section doesn't work yet and I already spent too much time trying to troubleshoot so leaving it for now.", color="#8ef", border="1px dashed red"))
    uploaded_files = st.file_uploader("Choose a file")

    model = whisper.load_model("small")