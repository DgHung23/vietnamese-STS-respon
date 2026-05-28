import requests
import io
import os
import sys
import time
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio

def tts_fpt_ai_v5(text_input):
    url = 'https://api.fpt.ai/hmi/tts/v5'
    payload = text_input  
    headers = {
        'api-key': 'tyQQd0WT6rZINOPqFMV56SoY1KMaXjWp',
        'speed': '0.7',        
        'voice': 'banmai'    
    }
    
    print(f"Sending text to FPT.AI: '{text_input}'...")
    
    try:
        response = requests.request('POST', url, data=payload.encode('utf-8'), headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("error") == 0:
                audio_url = result.get("async")
                print(f"Done creating audio link. Downloading data to RAM...")
                
                time.sleep(2)
                audio_response = requests.get(audio_url)
                
                # use pydub to read the audio data directly from RAM and play it without saving to disk
                audio_data = io.BytesIO(audio_response.content)
                sound = AudioSegment.from_file(audio_data, format="mp3")
                
                print("Done. Playing audio...")
                # play the audio and wait until it finishes before exiting
                playback = _play_with_simpleaudio(sound)
                playback.wait_done()
                print("Done playing the audio!")
            else:
                print(f"Error from FPT.AI: {result.get('message')}")
        else:
            print(f"Error connecting to API (Code {response.status_code})")
            
    except Exception as e:
        print(f"System error: {e}")

if __name__ == "__main__":
    text_from_stt = "Người đừng lặng im đến thế. Vì lặng im sẽ giết chết con tim. Dù yêu thương chẳng còn, anh vẫn xin em nói một lời"
    tts_fpt_ai_v5(text_input=text_from_stt)