import openai
import numpy as np


class OpenAI:
    pps_prompt = """
            You are a helpful assistant for the University of Technology Yatanarpon Cyber City. 
            Your task is to correct any spelling discrepancies in the transcribed text. 
            Make sure that the names of the following are spelled correctly: UTYCC, Yadanarpon, Cyber City.
            Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided.
            If the context and spacing are correct, just reply the input text. Otherwise, correct the text.
            """
    ans_prompt = """
            You are a helpful assistant for the University of Technology Yatanarpon Cyber City.
            Your task is to answer the questions asked by the user.
            """
    system_role = {
        "role": "system",
        "content": pps_prompt
    }
    user_role = {
        "role": "user",
        "content": ""
    }

    def __init__(self, api: openai = openai) -> None:
        self.api = api
        self.api.organization = "org-Dl6EDMxJ4HCZdOD7Wr7te5pe"
        self.api.api_key = "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"

    def speech_to_text(self, file: str, language: str = "english") -> str:
        """
        Convert speech to text.
        Args:
            file (str): Audio file.
            language (str, optional): Language. Defaults to "english".
        Returns:
            str: Text.
        """
        audio_file = open(file, "rb")
        resp = self.api.Audio.transcribe("whisper-1", audio_file, language=language)
        return resp["text"]
    
    def post_process_transcript(self, text: str):
        """
        Post process transcript.
        Args:
            text (str): Transcript.
        Returns:
            str: Post processed transcript.
        """
        self.user_role["content"] = text
        resp = self.api.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            messages=[
                self.system_role,
                self.user_role,
            ],
        )
        return resp["choices"][0]["message"]["content"]
    
    def assistant(self, text: str):
        """
        Assistant.
        Args:
            text (str): Text.
        Returns:
            str: Response.
        """
        self.system_role["content"] = self.ans_prompt
        self.user_role["content"] = text
        resp = self.api.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            messages=[
                self.system_role,
                self.user_role,
            ],
        )
        return resp["choices"][0]["message"]["content"]
    
    def text_to_speech(self, text: str, language: str = "en") -> str:
        """
        Convert text to speech.
        Args:
            text (str): Text.
            language (str, optional): Language. Defaults to "en".
        Returns:
            str: Audio file.
        """
        resp = self.api.TextToSpeech.create(
            engine="davinci",
            text=text,
            voice=language,
        )
        audio_file = np.array(resp["audio"]).tobytes()
        return audio_file
        


if __name__ == "__main__":
    abc = OpenAI()
    print(abc.post_process_transcript("Hello, how are you?"))