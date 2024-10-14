import numpy as np
from openai import OpenAI
from pydub import AudioSegment
from google.cloud import speech, texttospeech
from google.oauth2 import service_account

import warnings
warnings.filterwarnings("ignore")

class ConservationalAI:
    system_role = {"role": "system", "content": "You are a helpful assistant for the University."}
    user_role = {"role": "user", "content": ""}

    def __init__(self):
        client_file = "assets/smh-thesis-gcp.json"
        credentials = service_account.Credentials.from_service_account_file(client_file)
        self.speech_to_text_client = speech.SpeechClient(credentials=credentials)
        self.text_to_speech_client = texttospeech.TextToSpeechClient(credentials=credentials)
        self.chat_completion_client = OpenAI(api_key="")

    def speech_to_text(self, file: str, language = "en-US"):
        content = self.preprocess_audio(file)
        audio = speech.RecognitionAudio(content=content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=language,
        )
        response = self.speech_to_text_client.recognize(config=config, audio=audio)
        text = ""
        for result in response.results:
            text += result.alternatives[0].transcript

        return text
    
    def chat_completion(self, text: str, model: str = "gpt-4o-mini"): # "gpt-4o-mini"
        self.user_role["content"] = text
        resp = self.chat_completion_client.chat.completions.create(
            model=model,
            messages=[
                self.system_role,
                self.user_role,
            ],
        )
        return resp.choices[0].message.content
    
    def text_to_speech(self, text, language, output = None):
        if language == "my-MM":
            language = "en-US"
        synth_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )
        response = self.text_to_speech_client.synthesize_speech(
            input=synth_input, voice=voice, audio_config=audio_config
        )
        output_file = output if output else "output.wav"
        with open(output_file, "wb") as out:
            out.write(response.audio_content)
        return output_file
    
    def supported_languages(self):
        languages = {
            "english": "en-US",
            "japan": "ja-JP", 
            "korea": "ko-KR", 
            "chinese": "cmn-Hans-CN",
            "thai": "th-TH",
            "burmese": "my-MM",
        }
        return languages

    @staticmethod
    def preprocess_audio(file: str, sample_rate: int = 16000) -> np.ndarray:

        file_format = file.split(".")[-1]
        with open(file, "rb") as f:
            sound = AudioSegment.from_file(f, format=file_format)
            sound = sound.set_frame_rate(sample_rate)
            return sound.raw_data


if __name__ == "__main__":
    gcs = ConservationalAI()
    text = "What is your name?"
    answer = gcs.chat_completion(text)
    print(answer)

    
    