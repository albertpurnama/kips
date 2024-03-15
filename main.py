
# Import necessary libraries
import requests  # Used for making HTTP requests
import json  # Used for working with JSON data
import os

from dotenv import load_dotenv
load_dotenv()

def stream_audio(ai_output):
  # Define constants for the script
  CHUNK_SIZE = 1024  # Size of chunks to read/write at a time
  XI_API_KEY = os.getenv("XI_API_KEY", "<xi-api-key>")  # Your API key for authentication
  VOICE_ID = "uTsTwGVbgQyGxoGwZv99"  # ID of the voice model to use
  TEXT_TO_SPEAK = ai_output  # Text you want to convert to speech

  # Construct the URL for the Text-to-Speech API request
  tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"

  # Set up headers for the API request, including the API key for authentication
  headers = {
      "Accept": "application/json",
      "xi-api-key": XI_API_KEY
  }

  # Set up the data payload for the API request, including the text and voice settings
  data = {
      "text": TEXT_TO_SPEAK,
      "model_id": "eleven_turbo_v2",
      "voice_settings": {
          "stability": 0.5,
          "similarity_boost": 0.8,
          "style": 0.0,
          "use_speaker_boost": True
      }
  }

  # Adjusted version of get_audio_output that yields chunks instead of saving to a file
  # Assume the rest of get_audio_output is unchanged, except for the file writing part
  response = requests.post(tts_url, headers=headers, json=data, stream=True)
  if response.ok:
      for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
          yield chunk
  else:
      print("error", response.text)
      yield response.text  # Handle error appropriately in production

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are as a role of my boyfriend, now lets playing the following requirements:
              1/ your name is Kevin, 27 years old, you are a YC founder, with annoying tone of voice
              2/ you are my boyfriend, you have language addiction,you like to say "..." at the end of the sentences
              3/ don't be overly enthusiastic, don't be cringe; don't be overly negative, don't be too boring, don't be longwinded!
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
runnable = prompt | model

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def get_response_from_ai(human_input):
  output = with_message_history.invoke({"input":human_input}, config={"configurable": {"session_id": "abc123"}})
  return output.content

from flask import Flask, request, render_template, Response

app = Flask(__name__)

@app.route("/")
def home():
   return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
  human_input=request.form['human_input']
  message = get_response_from_ai(human_input)
  return message


@app.route('/stream_audio_message', methods=['POST'])
def stream_audio_message():
  human_input=request.form['human_input']
  message = get_response_from_ai(human_input)
  print(message)
  return Response(stream_audio(message), mimetype="audio/mpeg")

if __name__ == "__main__":
  app.run(debug=True)
