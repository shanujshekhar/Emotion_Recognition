import os
import tkinter
from tkinter import *
from PIL import ImageTk, Image
from threading import Thread
from pynput.keyboard import Key, Controller

def loadNews():
    
    global images,panel
    
    path="news"

    recordExp['state'] = 'normal'
    stop['state'] = 'normal'

    for filename in os.listdir(path):
        img = os.path.join(path,filename)
        if img is not None:
            images.append(img)
    
    readnews['state'] = 'disabled'
    nextnews['state'] = 'normal'
    
    nextNews()

def nextNews():
    
    global count,panel
    
    path = images[count%(len(images))]
    count += 1
    img2 = Image.open(path)
    width,height = img2.size
    img3 = img2.resize((int(width/4),int(height/4)))
    img = ImageTk.PhotoImage(img3)
    panel.configure(image=img)
    panel.image = img


def startReadingNews():
    global t1

    t1.start()

def startExp():
    global t2
    
    t2.start()


def expression():
    
    # recordExp.configure(state=DISABLED)
    os.system('python Ensemble_Final_Program.py')

def stopExp():

    keyboard = Controller()
    keyboard.press(Key.esc)
    keyboard.release(Key.esc)

t1 = Thread(target=loadNews)
t2 = Thread(target=expression)

count = 1
images = []

news = tkinter.Tk(screenName=None,baseName=None,className='readNews',useTk=1)
news.geometry("400x600")

img = ImageTk.PhotoImage(Image.open("news/news-1471859267.jpg"))

panel = tkinter.Label(news, image = img,width=500,height=500)
nextnews = tkinter.Button(news,text='next news',command=nextNews)
readnews = tkinter.Button(news,text='read news',command=startReadingNews)
recordExp = tkinter.Button(news,text='Record Expression',command=startExp)
stop = tkinter.Button(news,text='Stop Recording',command=stopExp)

nextnews.configure(state=DISABLED)
recordExp.configure(state=DISABLED)
stop.configure(state=DISABLED)

panel.pack(side="bottom",expand=1,fill="both")
readnews.pack(side="left",fill="x",expand=1)
nextnews.pack(side="right",fill="x",expand=1)
recordExp.pack(side="left")
stop.pack(side="right")

news.mainloop()
