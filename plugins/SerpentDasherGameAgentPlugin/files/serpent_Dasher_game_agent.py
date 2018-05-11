from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey
import keyboard
from . import ImageNetwork

class SerpentDasherGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

    def setup_play(self):

        context_classifier_path = f"datasets/context_classifier.model"

        context_classifier = ImageNetwork.ImageNetwork(
            input_shape=(480, 640, 3))  # Replace with the shape (rows, cols, channels) of your captured context frames

        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)

        self.machine_learning_models["context_classifier"] = context_classifier

    def handle_play(self, game_frame):
        print('Space is spressed: ',str(keyboard.is_pressed('space')))

        # for i, game_frame in enumerate(self.game_frame_buffer.frames):
        #     self.visual_debugger.store_image_data(
        #         game_frame.frame,
        #         game_frame.frame.shape,
        #         str(i)
        #     )
        # self.input_controller.tap_key(KeyboardKey.KEY_UP)
        eightframe = game_frame#.grayscale_frame#.eighth_resolution_frame
        context = self.machine_learning_models["context_classifier"].predict(eightframe.frame)
        if (context == 'jump') :
            self.input_controller.tap_key(KeyboardKey.KEY_UP)
        print(context)