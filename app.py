import gradio as gr
from audo import ConservationalAI

class GradioApp:
    def __init__(self):
        self.conservational_ai = ConservationalAI()
        self.languages = self.conservational_ai.supported_languages()

    def process(self, file, language):
        language = self.languages[language]
        text = self.conservational_ai.speech_to_text(file, language)
        answer = self.conservational_ai.chat_completion(text)
        answer_file = self.conservational_ai.text_to_speech(answer, language)
        return text, answer, answer_file

    def create_interface(self):
        iface = gr.Interface(
            fn=self.process,
            title="Conversational AI 🚀",
            inputs=[
                gr.Microphone(
                    label="Audio",
                    show_label=False,
                    type="filepath",
                ),
                gr.Dropdown(
                    choices=list(self.languages.keys()),
                    value="english",
                    label="Language"
                )
            ],
            outputs=[
                gr.Textbox(label="Question", lines=3),
                gr.Textbox(label="Answer", lines=5),
                gr.Audio(label="Answer Audio", type="filepath"),
            ],
        )
        return iface

if __name__ == "__main__":
    ai = GradioApp()
    iface = ai.create_interface()
    iface.queue().launch()