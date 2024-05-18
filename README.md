<!DOCTYPE html>
<html lang="en">
<body>

<h1>Face Recognition Project</h1>

<p>This project demonstrates a face recognition system using OpenCV, SQLite, and the Haar Cascade classifier for detecting and recognizing faces. The project includes three main components:</p>

<ol>
    <li><strong>Dataset Creation:</strong> Capturing face images and storing them in a dataset.</li>
    <li><strong>Training:</strong> Training the face recognizer on the captured dataset.</li>
    <li><strong>Face Detection and Recognition:</strong> Detecting and recognizing faces in real-time using the trained model.</li>
</ol>

<h2>Requirements</h2>
<ul>
    <li>Python 3.x</li>
    <li>OpenCV</li>
    <li>NumPy</li>
    <li>SQLite3</li>
    <li>PIL (Python Imaging Library)</li>
</ul>

<h2>Installation</h2>
<ol>
    <li>Clone the repository:
        <pre><code>git clone https://github.com/r-aalsan-jaas/Face-Detection-and-Recognition.git
cd facerecognition
</code></pre>
    </li>
    <li>Install the required Python packages:
        <pre><code>pip install opencv-python numpy pillow
</code></pre>
    </li>
</ol>

<h2>Usage</h2>

<h3>1. Dataset Creation</h3>
<p>This script captures images from your webcam and stores them in the <code>dataset</code> directory.</p>

<pre><code>import cv2
import numpy as np
import sqlite3

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

def insertorupdate(Id, Name, age):
    conn = sqlite3.connect("sqlite.db")
    cmd = "SELECT * FROM STUDENTS WHERE ID=" + str(Id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if isRecordExist == 1:
        conn.execute("UPDATE STUDENTS SET Name=?, age=? WHERE Id=?", (Name, age, Id))
    else:
        conn.execute("INSERT INTO STUDENTS (Id, Name, age) values(?,?,?)", (Id, Name, age))
    conn.commit()
    conn.close()

Id = input('Enter User Id:')
Name = input('Enter User Name:')
age = input('Enter User Age:')
insertorupdate(Id, Name, age)

sampleNum = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite("dataset/user." + str(Id) + "." + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitKey(100)
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if sampleNum > 50:
        break

cam.release()
cv2.destroyAllWindows()
</code></pre>

<h3>2. Training</h3>
<p>This script trains the face recognizer on the dataset and saves the trained model.</p>

<pre><code>import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "dataset"

def get_images_with_id(path):
    images_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for single_image_path in images_paths:
        faceImg = Image.open(single_image_path).convert('L')
        faceNp = np.array(faceImg, np.uint8)
        id = int(os.path.split(single_image_path)[-1].split(".")[1])
        print(id)
        faces.append(faceNp)
        ids.append(id)
        cv2.imshow("Training", faceNp)
        cv2.waitKey(100)
    return np.array(ids), faces

ids, faces = get_images_with_id(path)
recognizer.train(faces, ids)
recognizer.save("recognizer/trainingdata.yml")
cv2.destroyAllWindows()
</code></pre>

<h3>3. Face Detection and Recognition</h3>
<p>This script detects and recognizes faces in real-time using the webcam and displays the name and age of the recognized person.</p>

<pre><code>import cv2
import numpy as np
import sqlite3

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdata.yml")

def getprofile(id):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE id=?", (id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        profile = getprofile(id)
        if profile is not None:
            cv2.putText(img, "Name:" + str(profile[1]), (x, y+h+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
            cv2.putText(img, "Age:" + str(profile[2]), (x, y+h+45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
    cv2.imshow("FACE", img)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
</code></pre>

<h2>Haar Cascade Classifier</h2>
<p>The <code>haarcascade_frontalface_default.xml</code> is a pre-trained classifier used to detect faces. Ensure you have this file in your project directory.</p>

<h2>Project Structure</h2>
<pre><code>.
├── dataset                    # Directory to store captured face images
├── recognizer                 # Directory to store the trained model
├── sqlite.db                  # SQLite database file
├── haarcascade_frontalface_default.xml   # Haar Cascade Classifier file
├── dataset_creation.py        # Script for dataset creation
├── training.py                # Script for training the face recognizer
├── face_detection.py          # Script for face detection and recognition
└── README.md                  # Project README file
</code></pre>

<h2>References</h2>
<ul>
    <li><a href="https://docs.opencv.org/">OpenCV Documentation</a></li>
    <li><a href="https://www.sqlite.org/docs.html">SQLite Documentation</a></li>
</ul>

</body>
</html>
