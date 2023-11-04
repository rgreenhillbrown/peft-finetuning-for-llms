from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from threading import Thread

# Importing the backend logic
# Note: Ensure that the modules and functions are correctly defined in your project
from training_module import run_training  
from inference_module import run_inference
import tkinter as tk
from tkinter import filedialog

# Kivy String
kv = '''
ScreenManager:
    MainScreen:
        name: 'main'
    TrainScreen:
        name: 'train_screen'
    InferenceScreen:
        name: 'inference_screen'

<MainScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: [50, 50, 50, 50]
        spacing: 20

        Label:
            text: 'Welcome to the Model Training and Inference App!'
            font_size: '24sp'
            bold: True
            color: [0.3, 0.3, 0.3, 1]
        
        Label:
            text: 'Please select an option below to proceed:'
            font_size: '18sp'
            color: [0.5, 0.5, 0.5, 1]
        
        Button:
            text: 'Train'
            font_size: '20sp'
            background_color: [0.2, 0.6, 0.8, 1]
            color: [1, 1, 1, 1]
            size_hint_y: None
            height: '50sp'
            on_release: 
                root.manager.transition.direction = 'left'
                root.manager.current = 'train_screen'
        
        Button:
            text: 'Inference'
            font_size: '20sp'
            background_color: [0.8, 0.6, 0.2, 1]
            color: [1, 1, 1, 1]
            size_hint_y: None
            height: '50sp'
            on_release: 
                root.manager.transition.direction = 'left'
                root.manager.current = 'inference_screen'

<TrainScreen>:
    BoxLayout:
        orientation: 'vertical'
        Button:
            text: '←'
            size_hint: (None, None)
            width: '50sp'
            height: '50sp'
            on_release: root.go_back()

        Label:
            text: 'Select Source for Model'

        Spinner:
            id: model_source_spinner
            text: 'Local Storage'
            values: ('Local Storage', 'Huggingface Hub')
            on_text: root.update_view()

        BoxLayout:
            id: hub_box
            orientation: 'horizontal'
            TextInput:
                id: model_name_input
                hint_text: 'Enter model name'
            TextInput:
                id: token_input
                hint_text: 'Enter token'

        BoxLayout:
            id: local_box
            orientation: 'horizontal'
            size_hint_y: None
            height: '50sp'
            TextInput:
                id: file_path_input
                hint_text: 'Selected model file path'
                readonly: True
            Button:
                text: 'Browse'
                on_release: root.browse_file()

        
        Label:
            text: 'Fine-Tuning Parameters'
            font_size: '20sp'
            bold: True
        
        TextInput:
            id: learning_rate_input
            hint_text: 'Enter learning rate'
            multiline: False
        
        Button:
            text: 'Start Training'
            font_size: '20sp'
            size_hint_y: None
            height: '50sp'
            on_release: root.start_training()
        
        Label:
            id: train_status
            text: ''
            color: [0.8, 0, 0, 1]  # Color for error messages

<InferenceScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: [50, 50, 50, 50]
        spacing: 20

        AnchorLayout:
            anchor_x: 'left'
            anchor_y: 'top'
            
            Button:
                text: '←'
                size_hint: (None, None)
                width: '50sp'
                height: '50sp'
                pos_hint: {'top': 1}
                on_release: root.go_back()

            BoxLayout:
                orientation: 'vertical'
                padding: [50, 50, 50, 50]
                spacing: 20
                size_hint: 1, 1

        Label:
            text: 'Enter PEFT Model ID and Input Text for inference'
            font_size: '20sp'
            bold: True
        
        TextInput:
            id: peft_model_id_input
            hint_text: 'Enter PEFT Model ID'
            multiline: False
        
        TextInput:
            id: input_text_input
            hint_text: 'Enter input text'
            multiline: False
        
        Button:
            text: 'Perform Inference'
            font_size: '20sp'
            size_hint_y: None
            height: '50sp'
            on_release: root.perform_inference()
        
        Label:
            id: inference_status
            text: ''
            color: [0.8, 0, 0, 1]  # Color for error messages

'''

class MainScreen(Screen):
    pass

class TrainScreen(Screen):
    selected_file = ''

    def update_view(self):
        if self.ids.model_source_spinner.text == 'Huggingface Hub':
            self.ids.hub_box.opacity = 1
            self.ids.hub_box.disabled = False
            self.ids.local_box.opacity = 0
            self.ids.local_box.disabled = True
        else:
            self.ids.hub_box.opacity = 0
            self.ids.hub_box.disabled = True
            self.ids.local_box.opacity = 1
            self.ids.local_box.disabled = False

    def browse_file(self):
        root = tk.Tk()
        file_path = filedialog.askopenfilename(title="Select a Model File", filetypes=[("Model files", "*.json *.safetensors")])
        if file_path:
            self.selected_file = file_path[0]
            self.ids.file_path_input.text = self.selected_file

    def go_back(self):
        self.manager.transition.direction = 'right'
        self.manager.current = 'main'

    def start_training(self):
        source = self.ids.model_source_spinner.text
        if source == 'Huggingface Hub':
            model_name = self.ids.model_name_input.text
            token = self.ids.token_input.text
            # Handle the Huggingface Hub logic
        else:
            selected_files = self.ids.filechooser.selection
            if selected_files:
                local_model_path = selected_files[0]
                # Handle the local model logic using local_model_path

    def run_training_thread(self, model_name, token):
        try:
            run_training(model_name, token)
            self.ids.train_status.text = "Training completed!"
        except Exception as e:
            self.ids.train_status.text = f"Error: {str(e)}"

class InferenceScreen(Screen):
    def go_back(self):
        self.manager.transition.direction = 'right'
        self.manager.current = 'main'

    def perform_inference(self):
        peft_model_id = self.ids.peft_model_id_input.text
        input_text = self.ids.input_text_input.text
        if peft_model_id and input_text:
            self.ids.inference_status.text = "Performing inference..."
            Thread(target=self.run_inference_thread, args=(peft_model_id, input_text)).start()
            self.ids.peft_model_id_input.text = ""
            self.ids.input_text_input.text = ""

    def run_inference_thread(self, peft_model_id, input_text):
        try:
            result = run_inference(peft_model_id, input_text)
            self.ids.inference_status.text = f"Result: {result}"
        except Exception as e:
            self.ids.inference_status.text = f"Error: {str(e)}"

class MyApp(App):
    def build(self):
        return Builder.load_string(kv)

if __name__ == '__main__':
    MyApp().run()
