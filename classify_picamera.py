# python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Kevin Choi
# Modified the image classification example code to apply on our system

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import numpy as np
import time
import picamera
import pickle
import RPi.GPIO as GPIO
import sys
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from os import path

from PIL import Image
from tflite_runtime.interpreter import Interpreter

# load the object labels
def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}

# set up the tensor with the input model
def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

class Sheets_Logging:
   # The ID and range of a sample spreadsheet.
   SPREADSHEET_ID = '1GgUnSLkJ0ZnE35DLGpjPxCw3dhas-PeYQdw5oHVazlU'
   RANGE_NAME = 'Sheet1'
   # If modifying these scopes, delete the file token.pickle.
   SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

   def __init__(self):
       self.service = None
       self.credentials = self.auth()

   def auth(self):
       """Shows basic usage of the Sheets API.
       Prints values from a sample spreadsheet.
       """
       creds = None
       # The file token.pickle stores the user's access and refresh tokens, and is
       # created automatically when the authorization flow completes for the first
       # time.
       if path.exists('token.pickle'):
           with open('token.pickle', 'rb') as token:
               creds = pickle.load(token)
       # If there are no (valid) credentials available, let the user log in.
       if not creds or not creds.valid:
           if creds and creds.expired and creds.refresh_token:
               creds.refresh(Request())
           else:
               flow = InstalledAppFlow.from_client_secrets_file(
                   'credentials.json', self.SCOPES)
               creds = flow.run_local_server(port=0)
           # Save the credentials for the next run
           with open('token.pickle', 'wb') as token:
               pickle.dump(creds, token)
       self.service = build('sheets', 'v4', credentials=creds)

   def read_data(self):
       # Call the Sheets API
       service = self.service
       sheet = service.spreadsheets()
       result = sheet.values().get(spreadsheetId=self.SPREADSHEET_ID,
                                   range=self.RANGE_NAME).execute()
       values = result.get('values', [])
       if not values:
           print('No data found.')
           return None
       else:
           return values

   def write_data(self, data):
       service = self.service
       values = [data]
       body = {
           'values': values
       }
       range_name = 'Sheet1'
       result = service.spreadsheets().values().append(
           spreadsheetId=self.SPREADSHEET_ID, range=range_name,
           valueInputOption='USER_ENTERED', body=body).execute()

def main():
    doc = Sheets_Logging()
    # Set up the analog signal sent to the buzzer
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.OUT)

    # Read the data, find any expired item and alert the user if there is
    values = doc.read_data()
    cur_time = datetime.now()
    is_expired = 0
    for i, val in enumerate(values):
        if i:
            exp_date = datetime.strptime(val[2], '%Y-%m-%d %H:%M:%S')
            if exp_date < cur_time:
                is_expired = 1
                print(val) 
                break
    if is_expired:
        GPIO.output(17, True)
        time.sleep(5)
        GPIO.output(17, False)

    # Load the model and the labels, set the corresponding expiration dates
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', help='File path of .tflite file.', required=True)
    parser.add_argument(
        '--labels', help='File path of labels file.', required=True)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    expiration_dates = {
            "broccoli":       timedelta(days=5), 
            "cauliflower":    timedelta(weeks=1), 
            "zucchini":       timedelta(days=5), 
            "cucumber":       timedelta(days=10), 
            "artichoke":      timedelta(days=6), 
            "bell pepper":    timedelta(weeks=1), 
            "mushroom":       timedelta(weeks=1),
            "Granny Smith":   timedelta(weeks=4),
            "strawberry":     timedelta(days=5),
            "orange":         timedelta(weeks=4),
            "lemon":          timedelta(weeks=4),
            "pineapple":      timedelta(days=4),
            "banana":         timedelta(days=2),
            "pomegranate":    timedelta(weeks=4),
            "meat loaf":      timedelta(days=5),
            "pizza":          timedelta(days=3),
            "potpie":         timedelta(days=3),
            "burrito":        timedelta(days=3),
            "red wine":       timedelta(days=3)
            }
    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    # Regonize the image from the cimera, validate and write to the databse
    with picamera.PiCamera() as camera:
        camera.resolution = (width, height)
        camera.start_preview()
        count = 0
        prev_label_id = 0
        while True:
            stream = io.BytesIO()
            try:
                camera.capture(stream, format='jpeg')
                image = Image.open(stream).convert('RGB')
                results = classify_image(interpreter, image)
                label_id, prob = results[0]
                stream.seek(0)
                stream.truncate()
                print(labels[label_id], prob)
                if prob > .8 and label_id == prev_label_id:
                    count += 1
                    if count > 4:
                        if labels[label_id] in expiration_dates.keys():
                            # Valid for 5 consecutive times with high accuracy, and is a food item
                            doc.write_data([str(datetime.now()).split('.')[0], labels[label_id], str(datetime.now()+expiration_dates[labels[label_id]]).split('.')[0]])
                            GPIO.output(17, True)
                            time.sleep(5)
                            GPIO.output(17, False)
                        count = 0
                        print(labels[label_id] in expiration_dates.keys())
                elif label_id != prev_label_id:
                    count = 0
                prev_label_id = label_id
                time.sleep(.2)
            finally:
                pass

if __name__ == '__main__':
    main()
