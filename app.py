from flask import Flask, render_template, Response, jsonify
from camera import *

app = Flask(__name__)

headings = ("Name", "Video")

def music_rec():
    import pandas as pd
    data = {
        "Name": ["Name"],
        "Link": ["Link"],
    }
    return pd.DataFrame(data)

# Initialize DataFrame
df1 = music_rec().head(1)

@app.route('/')
def index():
    return render_template('index.html', headings=headings, data=df1)

def gen(camera):
    while True:
        global df1
        frame, df1 = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/t')
def gen_table():
    return df1.to_json(orient='records')

if __name__ == '__main__':
    app.debug = True
    app.run()
