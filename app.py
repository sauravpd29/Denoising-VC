import os
from re import S
import requests
import numpy as np
from PIL import Image
import streamlit as st
import soundfile as sf
from app_utils import model_denoising, play_file_uploaded, process_input_format
from config import *
import moviepy.editor as mp

import torch
from model_encoder import Encoder, Encoder_lf0
from model_decoder import Decoder_ac
from model_encoder import SpeakerEncoder as Encoder_spk
import os
import subprocess
from spectrogram import logmelspectrogram
import kaldiio
import resampy
import pyworld as pw



@st.cache(allow_output_mutation=True)
def load_session():
    return requests.Session()

def save_uploadedfile(uploadedfile, file_type):
    filename = uploadedfile.name

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    with open(os.path.join(UPLOAD_FOLDER, uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    
    process_input_format(filename, file_type)
    
def extract_logmel(wav_path, mean, std, sr=16000):
    # wav, fs = librosa.load(wav_path, sr=sr)
    wav, fs = sf.read(wav_path)
    if fs != sr:
        wav = resampy.resample(wav, fs, sr, axis=0)
        fs = sr
    #wav, _ = librosa.effects.trim(wav, top_db=15)
    # duration = len(wav)/fs
    assert fs == 16000
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
    mel = logmelspectrogram(
                x=wav,
                fs=fs,
                n_mels=80,
                n_fft=400,
                n_shift=160,
                win_length=400,
                window='hann',
                fmin=80,
                fmax=7600,
            )
    
    mel = (mel - mean) / (std + 1e-8)
    tlen = mel.shape[0]
    frame_period = 160/fs*1000
    f0, timeaxis = pw.dio(wav.astype('float64'), fs, frame_period=frame_period)
    f0 = pw.stonemask(wav.astype('float64'), f0, timeaxis, fs)
    f0 = f0[:tlen].reshape(-1).astype('float32')
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
    mean, std = np.mean(lf0[nonzeros_indices]), np.std(lf0[nonzeros_indices])
    lf0[nonzeros_indices] = (lf0[nonzeros_indices] - mean) / (std + 1e-8)
    return mel, lf0



    
    
    
def main():
    
    st.title('CAPSTONE PROJECT B33')
    st.subheader('🎵 Denoising')
    sess = load_session()
    uploaded_file = st.file_uploader("Upload audio here :", type=SUPPORT_FORMAT)
    # type = None
    file_type = ''
    file_name = ''
    file_format = ''
    # Play uploaded file
    if uploaded_file is not None:
        st.subheader('🔉 Uploaded audio')
        file_type = uploaded_file.type
        file_name = uploaded_file.name
        file_format = file_name[-3:]
        if file_format not in SUPPORT_FORMAT:
            st.error('Unsupported uploaded audio! (Try mp3 or wav)')
        else:
            play_file_uploaded(uploaded_file, file_type)

    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.beta_columns([1,1,1,1,1,2,1,1,1,1,1])  
    is_success=False

    if uploaded_file is not None and col6.button('Start Denoising 🎼'):
        # save file to backend
        save_uploadedfile(uploaded_file, file_type)
        m_amp_db, m_pha, pred_amp_db, X_denoise = model_denoising(file_name[:-3] + 'wav')
        is_success = True 
        st.success('Denoising Successful')
        
    if is_success:
        if 'audio' in uploaded_file.type:
            out_wav = file_name[:-3] + 'wav'
            out_audio_file = open(os.path.join(UPLOAD_FOLDER, f'out_{out_wav}'), 'rb')
            out_audio_bytes = out_audio_file.read()
            st.subheader('🔊 Denoised Output')
            st.audio(out_audio_bytes, format='audio/wav')
    
    
    
    
    
    st.subheader('🎵 Voice Conversion')

    #SOURCE VOICE
    source_voice = st.file_uploader("Upload source voice here  :", type=SUPPORT_FORMAT)
    file_type = ''
    file_name = ''
    file_format = ''
    if source_voice is not None:
        st.subheader('🔉 Uploaded source voice')
        file_type = source_voice.type
        file_name = source_voice.name
        file_format = file_name[-3:]
        if file_format not in SUPPORT_FORMAT:
            st.error('Unsupported uploaded audio! (Try mp3 or wav)')
        else:
            play_file_uploaded(source_voice, file_type)
            
            
    #TARGET VOICE
    target_voice = st.file_uploader("Upload target voice here  :", type=SUPPORT_FORMAT)
    file_type = ''
    file_name = ''
    file_format = ''
    if target_voice is not None:
        st.subheader('🔉 Uploaded target voice')
        file_type = target_voice.type
        file_name = target_voice.name
        file_format = file_name[-3:]
        if file_format not in SUPPORT_FORMAT:
            st.error('Unsupported uploaded audio! (Try mp3 or wav)')
        else:
            play_file_uploaded(target_voice, file_type)


    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.beta_columns([1,1,1,1,1,2,1,1,1,1,1])
    is_success2=False
    if source_voice is not None and target_voice is not None and col6.button('Convert 🎼'):
        # convert_examply.py
        source_fn=source_voice.name
        src_wav_path = './test_wavs/'+source_fn[:-3] + 'wav'
        target_fn=target_voice.name
        ref_wav_path = './test_wavs/'+target_fn[:-3] + 'wav'
        out_dir='./converted'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder = Encoder(in_channels=80, channels=512, n_embeddings=512, z_dim=64, c_dim=256)
        encoder_lf0 = Encoder_lf0()
        encoder_spk = Encoder_spk()
        decoder = Decoder_ac(dim_neck=64)
        encoder.to(device)
        encoder_lf0.to(device)
        encoder_spk.to(device)
        decoder.to(device)

        checkpoint_path = './checkpoints/useCSMITrue_useCPMITrue_usePSMITrue_useAmpTrue/VQMIVC-model.ckpt-500.pt' 
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(checkpoint["encoder"])
        encoder_spk.load_state_dict(checkpoint["encoder_spk"])
        decoder.load_state_dict(checkpoint["decoder"])

        encoder.eval()
        encoder_spk.eval()
        decoder.eval()
        
        mel_stats = np.load('./mel_stats/stats.npy')
        mean = mel_stats[0]
        std = mel_stats[1]
        feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=str(out_dir)+'/feats.1'))
        src_mel, src_lf0 = extract_logmel(src_wav_path, mean, std)
        ref_mel, _ = extract_logmel(ref_wav_path, mean, std)
        src_mel = torch.FloatTensor(src_mel.T).unsqueeze(0).to(device)
        src_lf0 = torch.FloatTensor(src_lf0).unsqueeze(0).to(device)
        ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)
        out_filename = os.path.basename(src_wav_path).split('.')[0] 
        with torch.no_grad():
            z, _, _, _ = encoder.encode(src_mel)
            lf0_embs = encoder_lf0(src_lf0)
            spk_emb = encoder_spk(ref_mel)
            output = decoder(z, lf0_embs, spk_emb)
            
            feat_writer[out_filename+'_converted'] = output.squeeze(0).cpu().numpy()
            feat_writer[out_filename+'_source'] = src_mel.squeeze(0).cpu().numpy().T
            feat_writer[out_filename+'_reference'] = ref_mel.squeeze(0).cpu().numpy().T
        
        feat_writer.close()
        
        cmd = ['parallel-wavegan-decode', '--checkpoint', \
               './vocoder/checkpoint-3000000steps.pkl', \
               '--feats-scp', f'{str(out_dir)}/feats.1.scp', '--outdir', str(out_dir)]
        subprocess.call(cmd)
        is_success2=True
        st.success('Conversion Successful')
        
        
    if is_success2:
        audio_file = open('./converted/'+out_filename+'_converted_gen.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
            
            
    st.subheader('Project details')
    my_expander1 = st.beta_expander('About Us')
    with my_expander1:
        st.text('Project Guide :  Prof. Swati\n\nTeam Members  :\nAayush Kapoor		PES2201800211\nSaurav Prasad		PES2201800098\nVihaan Pujara		PES2201800016\nAdya Manu		PES2201800108')




if __name__ == '__main__':

    st.set_page_config(
        page_title="Denoising and VC",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="collapsed",
        
    )
    
    
    main()
