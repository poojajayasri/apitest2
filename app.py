from flask import Flask, jsonify, request, render_template
from langchain.llms import OpenAI
from flask_cors import CORS
import os
import cv2
import tempfile
import NewTextClip

from moviepy.editor import *
import streamlit as stt
import openai
import requests
import time
import final3
from io import BytesIO
from elevenlabs import generate, set_api_key
from moviepy.editor import *
#import speech_recognition as sr
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy
from moviepy.editor import  AudioFileClip, vfx
import pysrt
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import os
#---------------------------
openai_api_key = os.environ.get("OPENAI_API_KEY")
set_api_key("5a27ae6995f4047b0d179989686e55c1")

llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)

app = Flask(__name__)
CORS(app)

# GET route
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
#------------------------------------------
# POST route
@app.route("/answer", methods=["POST", "GET"])
def answer():
    
    
    
    query = request.args.get("query")
    answer = llm(query)
    return jsonify({"answer": answer})
#------------------------------------------
if __name__ == "__main__":
    app.run(port=8000)
