import gradio as gr
from apis import OpenAI

languages = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "japanese": "ja",
    "korean": "ko",
    "russian": "ru",
}
def process(file, language):
    openai = OpenAI()
    print(f"[INFO] Language: {language}")
    language = languages[language]
    transcript = openai.speech_to_text(file, language)
    # transcript = openai.post_process_transcript(transcript)
    text = openai.assistant(transcript)
    print(f"[INFO] Finished: {text}")
    return transcript, text

iface = gr.Interface(
    fn=process,
    title="Conversational AI",
    inputs=[
        gr.Microphone(
            label="Audio",
            show_label=False,
            type="filepath",),
        gr.components.Dropdown(
        choices=["english", "spanish", "french", "german", "italian", "portuguese", "japanese", "korean", "russian"],
        value="english",
        label="Language")
        ],
    outputs=[
        gr.components.Textbox(label="Question", lines=3),
        gr.components.Textbox(label="Answer", lines=5),
    ]
)

iface.queue().launch(server_name="0.0.0.0")
    

# sed -i "s/\r//" backend/app/[filename].sh