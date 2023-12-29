import cv2
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from ultralytics import YOLO
from PIL import Image as PILImage

model = YOLO("gd.pt")

class MainScreen(Screen):
    pass

Builder.load_file("screen.kv")

class Main(App):

    def build(self):
        sm = ScreenManager()
        self.main_screen = MainScreen(name="main")
        sm.add_widget(self.main_screen)
        self.image = Image(
            size_hint=(None, None),
            size=(Window.width * 1, Window.height * 1),
            center=(Window.width * 0.5, Window.height * 0.5)
        )
        self.main_screen.add_widget(self.image, 1)

        # Carregar imagem da área de trabalho
        image_path = "pessoa.jpg"  # Substitua pelo caminho da sua imagem
        self.image_frame = cv2.imread(image_path)
        texture = self.convert_frame_to_texture(self.image_frame)
        self.image.texture = texture

        self.inference_count = 0
        Clock.schedule_interval(self.inference, 1 / 10)
        return sm

    def inference(self, *args):
        pixelated_frame = self.pixelate_frame(self.image_frame, 3)
        results = model.predict(source=pixelated_frame, verbose=False)
        People_count = len(results[0].boxes)
        print(f'Pessoas encontradas na inferência {self.inference_count + 1}: {People_count}')
        self.inference_count += 1
        App.get_running_app().stop()  # Finaliza a execução após 5 inferências

    def pixelate_frame(self, frame, block_size):
        small_frame = cv2.resize(frame, None, fx=1 / block_size, fy=1 / block_size, interpolation=cv2.INTER_NEAREST)
        pixelated_frame = cv2.resize(small_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        return pixelated_frame

    def convert_frame_to_texture(self, frame):
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        return texture

if __name__ == '__main__':
    Main().run()
