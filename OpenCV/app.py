# import the required libraries
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import json
import sys
import cv2
from tensorflow.keras.models import load_model
from string import ascii_lowercase
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
sys.path.append("..")
from Preprocess import preprocess



# open and get the constants from the json file
f = open('../Constants.json')
data = json.load(f)

labels = data['labels']

# number of classes
NUM_CLASSES = data['NUM_CLASSES']

# size of the images to be generated
IMG_SIZE = data['TRAIN_IMG_SIZE']

classes = ascii_lowercase + '0123456789'

# loading the weights for the model
model=load_model('../sign_lang.h5')


class GUI:
    
    def __init__(self):
        self.current_word = ''
        self.sentence = ''
        self.character = ''
        self.canvas = None
        
        self.cap = cv2.VideoCapture(0)
        self.root = Tk()
        
        self.root.title("Sign language to Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        
        self.count = {}
        self.blank_count = 0
        for i in classes:
            self.count[i] = 0
        
        self.build_ui()
        self.video_stream()
        
    def build_ui(self):
        
        # width*height
        self.root.geometry('1000x600')
        self.video_label = Label(self.root)
        self.video_label.grid(rowspan = 400, columnspan = 600)

        # Prediction Description Label
        self.predicted_desc = Label(self.root)
        self.predicted_desc.grid(row = 600, column = 10)
        self.predicted_desc.config(text = "Character :", font = ("Courier",20, "bold"))

        self.char = StringVar()

        # Prediction Label
        self.predicted = Label(self.root, textvariable = self.char)
        self.predicted.grid(row = 600, column = 60)
        self.predicted.config(textvariable = self.char,font = ("Courier", 20, "bold"))

        # Word Description Label
        self.word_desc = Label(self.root, text = 'Word:')
        self.word_desc.grid(row = 800, column = 10)
        self.word_desc.config(text = "Word:", font = ("Courier", 20, "bold"))


        self.word = StringVar()

        # Word Label
        self.word_formed = Label(self.root, textvariable = self.word)
        self.word_formed.grid(row = 800, column = 60)
        self.word_formed.config(textvariable = self.word, font = ("Courier", 20, "bold"))
        
        # Sentence Description Label
        self.sent_desc = Label(self.root)
        self.sent_desc.grid(row = 1000, column = 10)
        self.sent_desc.config(text = "Sentence:", font = ("Courier", 20, "bold"))


        self.sent = StringVar()

        # Sentence Label
        self.sentence_formed = Label(self.root, textvariable = self.sent)
        self.sentence_formed.grid(row = 1000, column = 60)
        self.sentence_formed.config(textvariable = self.sent, font = ("Courier", 20, "bold"))

    def updateDepositLabel(self, *args):
        self.char.set(args[0])
        self.word.set(args[1])
        self.sent.set(args[2])
        
    # function for video streaming
    def video_stream(self):
        
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)

        hand_img = preprocess(frame)
        probs_class_map, self.prediction = self.predict(hand_img)
        
        if len(self.current_word) != 0 and self.prediction == 'blank':
            self.blank_count += 1
            if self.blank_count > 80:
                self.sentence += ' '
                self.sentence += self.current_word
                self.current_word = ''
                for i in classes:
                    self.count[i] = 0
                self.blank_count = 0
                
        elif self.prediction != 'blank':
            self.count[self.prediction] += 1
            if self.count[self.prediction] > 50:
                self.current_word += self.prediction
                for i in classes:
                    self.count[i] = 0
                self.blank_count = 0
                self.plot(probs_class_map)
       
            
        self.updateDepositLabel(self.prediction, self.current_word, self.sentence)
        cv2Img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2Img)
        imgtk = ImageTk.PhotoImage(image = img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image = imgtk)
        self.video_label.after(1, self.video_stream)
    
    
    def destructor(self):
        self.root.destroy()
        self.cap.release()
        
    def predict(self,hand_img):
        resized_hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
        x_test = np.array(resized_hand_img)
        x_test = x_test.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        probs = model.predict(x_test)[0]
        pred = np.argmax(probs)
        probs_class_map = [(index, probs[index]) for index in np.argsort(probs)[-5:]]
        
        return probs_class_map, labels[str(pred)]
    
    def plot(self,probs_class_map):
        
        # destroying the old widget
        if(self.canvas):
            self.canvas.get_tk_widget().destroy()

        f = plt.Figure(figsize = (6,5), dpi = 60)
        ax = f.add_subplot(111)
        width = 0.5
        probs = [prob for curr_class,prob in probs_class_map]
        most_classes = [labels.get(str(curr_class),'blank') for curr_class,prob in probs_class_map]
        rects = ax.bar(most_classes,probs, width)
        ax.set_xlabel('Top 5 Classes')
        ax.set_ylabel('Probabilities')
        ax.set_title('Class Wise Probabilities')
        self.canvas = FigureCanvasTkAgg(f, master = self.root)
        self.canvas.get_tk_widget().grid(row = 300, column = 720)
        self.canvas.draw()


gui = GUI()
gui.root.mainloop()


gui.cap.release()