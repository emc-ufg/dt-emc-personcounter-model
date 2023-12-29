import cv2
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from ultralytics import YOLO

model = YOLO("gd.pt")

People_count = 0

class MainScreen(Screen):
    pass

Builder.load_file("screen.kv")

class Main(App):

    def build(self):
        #Aqui é apenas um gerenciamento de tela para adicionar os frames da camera à tela
        sm = ScreenManager()
        self.main_screen = MainScreen(name="main")
        sm.add_widget(self.main_screen)
        self.image = Image(
            size_hint=(None, None),
            size=(Window.width * 1, Window.height * 1),
            center=(Window.width * 0.5, Window.height * 0.5)
        )
        self.main_screen.add_widget(self.image, 1)
        #para a aplicacao mesmo nao vamos querer essas telas, mas por enquanto é legal para visualizar se a camera esta abrindo

        #aquin é onde definimos a forma de captura e os períodos (1/frequencia) de execucao de cada funcao
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1 / 60)
        Clock.schedule_interval(self.inference, 1 / 10)
        return sm

    def load_video(self, *args):
        ret, frame = self.capture.read()
        if ret:
            self.image_frame = frame
            buffer = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def inference(self, *args):
        pixelated_frame = self.pixelate_frame(self.image_frame, 3)
        results = model.predict(source=pixelated_frame, verbose=False, show=True)
        preds_pixelate, classes_pixelated = results[0].boxes.conf.cpu().numpy(), results[0].boxes.cls.cpu().numpy()
        try:
            if int(classes_pixelated[0]) == 0:
                if preds_pixelate[0] > 0.5:
                    People_count = len(results[0].boxes)  # Conta o número de caixas delimitadoras
                    print(f'Pessoas encontradas: {People_count}')
        except:
            pass

    def pixelate_frame(self,frame, block_size):
        small_frame = cv2.resize(frame, None, fx=1 / block_size, fy=1 / block_size, interpolation=cv2.INTER_NEAREST)
        pixelated_frame = cv2.resize(small_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        return pixelated_frame

if __name__ == '__main__':
    Main().run()
