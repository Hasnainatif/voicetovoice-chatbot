# Import libraries
import os
from groq import Groq
import whisper
from gtts import gTTS
import gradio as gr
model = whisper.load_model("base")
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)
# Define a single-step function for the chatbot
def voice_to_voice_chatbot(audio_input):
    # Step 1: Transcribe audio to text
    transcribed_text = model.transcribe(audio_input)["text"]

    # Step 2: Get response from LLM using Groq API
    response_text = client.chat.completions.create(
        messages=[{"role": "user", "content": transcribed_text}],
        model="llama3-8b-8192",
    ).choices[0].message.content

    # Step 3: Convert response text to audio
    tts = gTTS(text=response_text, lang='en')
    response_audio_path = "response.mp3"
    tts.save(response_audio_path)

    return response_audio_path

# Set up Gradio interface
interface = gr.Interface(
    fn=voice_to_voice_chatbot,
    inputs=gr.Audio(type="filepath"),  # Audio file input
    outputs=gr.Audio(type="filepath"),  # Audio file output
    title="Real-Time Voice-to-Voice Chatbot"
)

# Launch Gradio app
interface.launch(share=True)
