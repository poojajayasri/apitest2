from flask import Flask, jsonify, request, render_template
from langchain.llms import OpenAI
from flask_cors import CORS
import os
import cv2
import tempfile
import NewTextClip

from moviepy.editor import *
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
#openai_api_key = os.environ.get("OPENAI_API_KEY")
set_api_key("5a27ae6995f4047b0d179989686e55c1")

#llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)

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
    
    temp_dir_one = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_one.name
    print(temp_dir)
    class ImageGenerator:
        def __init__(self) -> str:
            self.image_urls = []
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            #"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"
            #os.environ["OPENAI_API_KEY"] = "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"
            self.name = None

        def generateImage(self, prompt, image_count, image_size):
            try:
                response = openai.Image.create(
                    prompt=prompt,
                    n=image_count,
                    size=image_size,
                )
                time.sleep(10)
                self.image_urls += [image["url"] for image in response['data']]
            except openai.error.OpenAIError as e:
                print(e.http_status)
                print(e.error)

        def downloadImages(self, prompt_index, folder):
            try:
                for i, url in enumerate(self.image_urls):
                    print(i)
                    print(url)
                    #image = requests.get(url)
                    try:
                        response = requests.get(url)
                        response.raise_for_status()  # raise an error if HTTP status code indicates a failure
                    except requests.exceptions.RequestException as e:
                        print(f"Error downloading image: {e}")
                        continue  # move on to the next image URL
                    with open(f"{temp_dir}/{prompt_index + 1}.jpg", "wb") as f:
                        f.write(response.content)
                    image = f"{temp_dir}/{prompt_index + 1}.jpg"

                    """
                    with open(f"{temp_dir}/{prompt_index + 1}.jpg", "wb") as f:
                        f.write(image.content)
                        print(here before the 
                        image = f.open(f"{temp_dir}/{prompt_index + 1}.jpg")
                    """
            except Exception as e:
                print(f"An error occurred: {e}fvdfvwef")

    #------------------------------------
    def zoom(img,ii):
        # Set the output video dimensions
        width, height = img.shape[1], img.shape[0]
        size = (width, height)
        fps = 30
        # Set the zoom parameters
        start_scale = 1.0
        end_scale = 2.0
        zoom_steps = 300  # Number of frames to zoom in/out

        out_file = f"{temp_dir}/output{ii}.mp4"
        # Create an output video writer
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        except Exception as e:
            # handle any other exception
            print(e)
            
        print("YOYOYOY")
        try:
            out = cv2.VideoWriter(out_file, fourcc, fps, size)
        except Exception as e:
            # handle any other exception
            print(f" twowowowo An error occurred: {e}")
        #print("this is outt!")
        print(out)
        # Add zoom effect to the image and write each frame to the output video
        for i in range(zoom_steps):
            scale = start_scale + (end_scale - start_scale) * (i / zoom_steps)
            zoomed_img = cv2.resize(img, None, fx=scale, fy=scale)
            x_offset = int((zoomed_img.shape[1] - width) / 2)
            y_offset = int((zoomed_img.shape[0] - height) / 2)
            cropped_img = zoomed_img[y_offset:y_offset + height, x_offset:x_offset + width]
            out.write(cropped_img)

        # Write the last frame multiple times to make the video 10 seconds long
        video_duration = 10  # seconds
        for i in range(int(fps * (video_duration - zoom_steps / fps))):
            out.write(cropped_img)

        # Release the video writer and destroy all windows
        out.release()
        with open(f"{temp_dir}/output{ii}.mp4", 'rb') as f:
                video_bytes = f.read()
        cv2.destroyAllWindows()
    #------------------------------------------
    def time_to_seconds(time_obj):
        return time_obj.hours * 3600 + time_obj.minutes * 60 + time_obj.seconds + time_obj.milliseconds / 1000


    def create_subtitle_clips(subtitles, videosize,fontsize=24, font='Arial', color='blue', debug = False):
        subtitle_clips = []

        for subtitle in subtitles:
            start_time = time_to_seconds(subtitle.start)
            end_time = time_to_seconds(subtitle.end)
            duration = end_time - start_time

            video_width, video_height = videosize

            text_clip = NewTextClip.TextClip(subtitle.text, fontsize=fontsize, color=color, bg_color = 'yellow',size=(round(video_width*3/4), 100)).set_start(start_time).set_duration(duration)

            #text_clip = TextClip(subtitle.text, fontsize=fontsize, font=font, color=color, bg_color = 'black',size=(video_width*3/4, None), method='caption').set_start(start_time).set_duration(duration)
            subtitle_x_position = 'center'
            subtitle_y_position = video_height* 4 / 5 

            text_position = (subtitle_x_position, subtitle_y_position)                    
            subtitle_clips.append(text_clip.set_position(text_position))

        return subtitle_clips
    #------------------------------------------
    def crossfade(ii, total_images):
        video_clips =[]
        clip1 = []
        for iii in range(total_images):
            clip1.append(VideoFileClip(f"{temp_dir}/output{iii+1}.mp4"))

        max_width = max(clip1[0].w, clip1[1].w)
        max_height = max(clip1[0].h, clip1[1].h)

        for i in range(0, total_images):
            clip1[i] = clip1[i].resize((max_width, max_height))
            video_clips.append(clip1[i])

        padding = 1.5

        video_fx_list = [video_clips[0]]

        idx = video_clips[0].duration - padding
        for video in video_clips[1:]:
            video_fx_list.append(video.set_start(idx).crossfadein(padding))
            idx += video.duration - padding

        final_video = CompositeVideoClip(video_fx_list)
        if(ii == total_images-1):

     #-----------------------------------------

            with open(f"{temp_dir}/narration.txt", "r") as f:
                text = f.read()


            audio_bytes = generate(
            text=text,
            voice="Bella",
            model="eleven_multilingual_v1"
            )

            audio_bytes_io = BytesIO(audio_bytes)
            with open(f"{temp_dir}/movieaudio.mp3", 'wb') as f:
               f.write(audio_bytes)


            audio_clip = AudioFileClip(f"{temp_dir}/movieaudio.mp3")
            video_clip = final_video
            duration_ratio = video_clip.duration / audio_clip.duration
            stretched_video = video_clip.fx(vfx.speedx, duration_ratio)
            final_clip = stretched_video.set_audio(audio_clip)


            video = final_clip

            #COMMFORDEV
            with open(f"{temp_dir}/movieaudio.mp3", "rb") as audio_file:
                transcript = openai.Audio.transcribe(
                    file = audio_file,
                    model = "whisper-1",
                    response_format="srt",
                    language="en"

                )

            with open(f"{temp_dir}/subz.srt", 'w') as f:
                f.write(transcript)

            srtfilename = f"{temp_dir}/subz.srt"
            subtitles = pysrt.open(srtfilename)

            subtitle_clips = create_subtitle_clips(subtitles,video.size)

            final_video = CompositeVideoClip([video] + subtitle_clips)
            clip = final_video.subclip(0, audio_clip.duration)
            clip.write_videofile(f"{temp_dir}/moviesubbd.mp4")



            with open(f"{temp_dir}/moviesubbd.mp4", 'rb') as f:
                video_bytes = f.read()
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------

    query = request.args.get("query")
    idea = query
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    a = "Dall e"
    if idea:
        #COMMFORTEST
        final3.define_idea(idea, temp_dir)
        time.sleep(20)
        if a == "Dall e":

            imageGen = ImageGenerator() 

            with open(f"{temp_dir}/ttt.txt", "r") as f:
                prompts = f.readlines()

            local_file_path =[]
            total_images = 3

            for i, prompt in enumerate(prompts):
                prompt = f"{i+1}.{prompt}"
                print(prompt)

                imageGen.generateImage(
                    prompt=prompt,
                    image_count=1,
                    image_size='1024x1024'
                )
                imageGen.downloadImages(prompt_index=i, folder='{temp_dir}')
            #--------------------------
            for i in range(1, 3):
                img1 = cv2.imread(f"{temp_dir}/{i}.jpg", -1)

                if img1 is None:
                    print(f"Error: Failed to read image file {temp_dir}/{i}.jpg")

                zoom(img1,i)
                print(i)

            # Release the video writer and destroy all windows

            cv2.destroyAllWindows()
            #-------------------------
            total_images = 2
            ii = total_images - 1
            crossfade(ii, total_images)



    return jsonify({"answer": answer})
#------------------------------------------
if __name__ == "__main__":
    app.run(port=8000)
