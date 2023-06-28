#Environment variables
import os
from dotenv import load_dotenv  

#LangChain
from langchain.llms import OpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

#Whisper
import whisper
import mimetypes
from pydub import AudioSegment
from moviepy.editor import VideoFileClip

#Streamlit
import streamlit as st
from streamlit_extras.buy_me_a_coffee import button


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


def get_audio_transcripts(file):
    model = whisper.Load_model("base")
    result = model.transcribe(file)
    return result["text"]

def handle_audio_file(file, file_path):
    # Determine audio format from file extension
    file_type = mimetypes.guess_type(file.name)[0]
    audio = None

    if file_type == "audio/mpeg":
        audio = AudioSegment.from_mp3(file_path)
    elif file_type == "audio/ogg":
        audio = AudioSegment.from_ogg(file_path)
    elif file_type == "audio/wav":
        audio = AudioSegment.from_wav(file_path)
    else:
        st.write(f"Unsupported file type: {file_type}")

    return audio

#Start Streamlit
#Remove hamburger menu and Streamlit watermark

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Set the Streamlit page configuration
st.set_page_config(page_title="Lu0's handy transcription app", page_icon="🐱", layout="wide")

# Start Top Information
st.title("Lu0's handy transcription app 🐱")
st.markdown("### An app I made to easily generate transcripts for videos.")

# End Top Information

# Output type selection by the user
output_type = st.radio(
    "Video Type:",
    ('Youtube video', 'Local file'), horizontal=True)


if output_type == 'Youtube video':
    youtube_video = st.text_input(label="Youtube URL",  placeholder="Ex: https://www.youtube.com/watch?v=dQw4w9WgXcQ", key="yt_video")
    transcripts = get_yt_transcripts(youtube_video)
    for i, (transcription, metadata) in enumerate(transcripts):
            st.text(f"Transcription for '{metadata['title']}'")
            st.image(metadata['thumbnail_url'], caption=metadata['title'])
            st.text_area(value=transcription, height=200, max_chars=None)


elif output_type == 'Local file':
    uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)

    # Maximum file size in bytes for different ranges
    MAX_FILE_SIZE_25MB = 25 * 1024 * 1024
    MAX_FILE_SIZE_50MB = 50 * 1024 * 1024
    MAX_FILE_SIZE_75MB = 75 * 1024 * 1024
    MAX_FILE_SIZE_100MB = 100 * 1024 * 1024

    if output_type == 'Local file':
        uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)

        if uploaded_files:
            for file in uploaded_files:
                # Check file size
                file_size = file.size

                if file_size > MAX_FILE_SIZE_25MB:
                    num_splits = 1
                    if file_size <= MAX_FILE_SIZE_50MB:
                        num_splits = 2
                    elif file_size <= MAX_FILE_SIZE_75MB:
                        num_splits = 3
                    elif file_size <= MAX_FILE_SIZE_100MB:
                        num_splits = 4
                    else:
                        st.write(f"File {file.name} is larger than 100MB. Cannot process files larger than 100MB.")
                        continue
                
                    st.write(f"File {file.name} is larger than 25MB. Splitting the file into {num_splits} parts...")

                    # Determine file type
                    file_type = mimetypes.guess_type(file.name)[0]
                
                    if "video" in file_type:
                        # Extract audio from video file
                        clip = VideoFileClip(file)
                        clip.audio.write_audiofile("audio.wav")
                        file_path = "audio.wav"
                    elif "audio" in file_type:
                        # Write audio file
                        with open("audio.wav", "wb") as audio_file:
                            audio_file.write(file.getbuffer())
                        file_path = "audio.wav"
                    else:
                        st.write(f"Unsupported file type: {file_type}")
                        continue

                    # Load the audio file
                    audio = handle_audio_file(file, file_path)

                    # Calculate the length of each split
                    split_length = len(audio) // num_splits
                    split_files = [audio[i * split_length: (i + 1) * split_length] for i in range(num_splits)]

                    # Save the split files and process each one separately
                    for index, split_file in enumerate(split_files):
                        split_filename = f"split{index + 1}.wav"
                        split_file.export(split_filename, format="wav")

                        transcription = get_audio_transcripts(split_filename)

                        st.text_area(f"Transcription for {file.name} - Part {index + 1}", value=transcription, height=200, max_chars=None)
                    
                    #Clean up the split audio file
                    os.remove(split_filename)

                    # Clean up the extracted audio file
                    os.remove(file_path)

                else:
                    transcription = get_audio_transcripts(file)
                    st.text_area(f"Transcription for {file.name}", value=transcription, height=200, max_chars=None)
