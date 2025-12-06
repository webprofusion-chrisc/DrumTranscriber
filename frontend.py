import time
from pytube import YouTube
from streamlit_player import st_player

from DrumTranscriber import DrumTranscriber
from utils.config import SETTINGS

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import tempfile

import librosa
import librosa.display
import pretty_midi


@st.cache_resource
def initialise_transcriber():
    transcriber = DrumTranscriber()
    return transcriber

transcriber = initialise_transcriber()

@st.cache_data
def process_youtube(link, start_from=0):
    yt = YouTube(link)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download(output_path=".")
    
    base, ext = os.path.splitext(out_file)
    new_file = base + '.wav'
    os.rename(out_file, new_file)
    
    try:
        samples, sr = librosa.load(
            new_file, sr=44100, offset=start_from)
        tempo, _ = librosa.beat.beat_track(y=samples, sr=sr)
        if hasattr(tempo, 'item'):
            tempo = tempo.item()
    finally:
        if os.path.exists(new_file):
            os.remove(new_file)

    preds = transcriber.predict(samples, sr)
    return preds, samples, sr, tempo

@st.cache_data
def process_uploaded_file(file_bytes, file_name, start_from=0):
    # Create a temp file to save the uploaded bytes
    suffix = os.path.splitext(file_name)[1]
    if not suffix:
        suffix = ".wav"
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
        
    try:
        samples, sr = librosa.load(
            tmp_path, sr=44100, offset=start_from)
        tempo, _ = librosa.beat.beat_track(y=samples, sr=sr)
        if hasattr(tempo, 'item'):
            tempo = tempo.item()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    preds = transcriber.predict(samples, sr)
    return preds, samples, sr, tempo

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def create_midi(preds, tempo):
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    drum_program = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    
    # Map labels to MIDI notes
    # 0: 'crash', 1: 'hihat_c', 2: 'kick_drum', 3: 'ride', 4: 'snare', 5: 'tom_h'
    midi_map = {
        'crash': 49,
        'hihat_c': 42,
        'kick_drum': 36,
        'ride': 51,
        'snare': 38,
        'tom_h': 50
    }

    for index, row in preds.iterrows():
        note_number = midi_map.get(row['prediction'])
        if note_number:
            # Create a note instance, assuming a short duration for drum hits (e.g., 0.1s)
            note = pretty_midi.Note(
                velocity=100, 
                pitch=note_number, 
                start=row['time'], 
                end=row['time'] + 0.1
            )
            drum_program.notes.append(note)

    midi_data.instruments.append(drum_program)
    
    # Write to a temporary file and read bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
        midi_data.write(tmp.name)
        tmp_path = tmp.name
        
    with open(tmp_path, "rb") as f:
        midi_bytes = f.read()
        
    os.remove(tmp_path)
    return midi_bytes

transcriber = initialise_transcriber()

st.title('Drum Transcriber Demo')

input_method = st.radio("Choose input method:", ["Audio File Upload", "YouTube Link"])

youtube_link = None
uploaded_file = None

if input_method == "YouTube Link":
    youtube_link = st.text_input('Input youtube link here',
                          value='https://www.youtube.com/watch?v=4SDBJp_B5qQ')
    if youtube_link:
        st_player(youtube_link)
else:
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

start_from = st.number_input(label='Start from (in seconds)', min_value=0)

preds = None
samples = None
sr = None
tempo = 120.0

if input_method == "YouTube Link" and youtube_link:
    st.title('Predictions')
    with st.spinner("Processing YouTube video..."):
        preds, samples, sr, tempo = process_youtube(youtube_link, start_from)
elif input_method == "Audio File Upload" and uploaded_file:
    st.title('Predictions')
    with st.spinner("Processing uploaded file..."):
        preds, samples, sr, tempo = process_uploaded_file(uploaded_file.getvalue(), uploaded_file.name, start_from)

if preds is not None:
    st.write(f"Detected Tempo: {tempo:.2f} BPM")
    # Play the processed audio snippet
    st.audio(samples, sample_rate=sr)

    labelled_preds = [SETTINGS['LABELS_INDEX'][i] for i in
                      np.argmax(
                          preds[SETTINGS['LABELS_INDEX'].values()].to_numpy(), axis=1)
                      ]

    preds['prediction'] = labelled_preds
    preds['confidence'] = preds.apply(
        lambda x: f"{x[x['prediction']]*100:.1f}%", axis=1)
    preds['time'] = preds['time'].round(2)

    st.write(preds[['time', 'prediction', 'confidence']].T)

    fig, ax = plt.subplots(sharex=True, nrows=7, figsize=(20, 20))

    librosa.display.waveshow(
        samples, sr=sr, offset=start_from, ax=ax[0])

    ax[0].set_yticklabels([])
    ax[0].set_xlabel(None)

    for i in range(1, 7):
        label_name = SETTINGS['LABELS_INDEX'][i-1]

        hit_times = np.array(
            preds[preds['prediction'] == label_name]['time'].to_list())

        ax[i].vlines(hit_times+start_from, -1, 1)
        ax[i].set_ylabel(label_name, rotation=0, fontsize=20)
        ax[i].set_yticklabels([])

        if i == 6:
            ax[i].set_xlabel('time')

    st.pyplot(fig)

    st.download_button(
        "Press to Download CSV",
        convert_df(preds),
        "predictions.csv",
        "text/csv",
        key='download-csv'
    )

    st.download_button(
        "Press to Download MIDI",
        create_midi(preds, tempo),
        "drum_transcription.mid",
        "audio/midi",
        key='download-midi'
    )

    st.dataframe(preds, use_container_width=True)
