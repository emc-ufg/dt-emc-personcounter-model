from flask import Flask, request
from flask_restful import Resource, Api
from flask_restful.reqparse import RequestParser
import cv2
import numpy as np
import base64
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from ultralytics import YOLO
import paho.mqtt.client as mqtt

app = Flask(__name__)
api = Api(app)
model = YOLO("gd.pt")

# Configurações do MQTT
mqtt_broker_address = "200.137.220.250"
mqtt_topic = "CAE/S101/sensor/pessoas"

class InferenceResource(Resource):
    def __init__(self):
        self.inference_count = 0
        self.image_frame = None
        self.image_texture = None

    def post(self):
        parser = RequestParser()
        parser.add_argument('image', type=str, required=True)
        args = parser.parse_args()

        image_data = args['image'].split(',')[1]  # Remove o cabeçalho da string base64
        image_decoded = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)

        # Generate a random number and character
        random_number = np.random.randint(0, 10)
        random_letter = chr(np.random.randint(97, 123))

        self.inference_count += 1
        self.image_frame = image_decoded

        # Realizar uma única inferência
        pixelated_frame = self.pixelate_frame(self.image_frame, 3)
        results = model.predict(source=pixelated_frame, verbose=False)
        people_count = len(results[0].boxes)
        print(f'Pessoas encontradas na inferência: {people_count}')

        # Concat the random number and letter to the image filename
        image_filename = f"received_image_{random_number}_{random_letter}.png"

        #cv2.imwrite(image_filename, image_decoded)
        #print(f'Imagem salva como: {image_filename}')

        # Enviar resultado para o tópico MQTT
        self.send_mqtt_message(people_count)

        return {'people_count': people_count}

    def convert_frame_to_texture(self, frame):
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        return texture

    def pixelate_frame(self, frame, block_size):
        small_frame = cv2.resize(frame, None, fx=1 / block_size, fy=1 / block_size, interpolation=cv2.INTER_NEAREST)
        pixelated_frame = cv2.resize(small_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        return pixelated_frame

    def send_mqtt_message(self, message):
        client = mqtt.Client()
        client.connect(mqtt_broker_address)
        client.publish(mqtt_topic, "{\"value\": " + str(message) + "}")
        client.disconnect()

api.add_resource(InferenceResource, '/inference')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9999)
