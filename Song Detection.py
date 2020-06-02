from tkinter import *
from tkinter import ttk
import tkinter
from PIL import Image, ImageTk
import os
from pygame import mixer
import random
import re
from threading import Thread
import time

choosenSong = 'Animals.mp3'
mixer.init()
mixer.music.load("PlayList/02 Animals.mp3")

root = Tk()
root.title("Genre Detection of Song")

lb = Label(root, text=choosenSong)
lb.pack(side="bottom", fill="x", expand=1)

def startThreads():
	
	global t1
	global t2
	global b0

	b0.config(state=DISABLED)

	t1.start()
	t2.start()

b0 = Button(root, text='Play & Record', bg="white", fg='black', bd=4, relief='raised', command=startThreads)

def nextSong():
	
	global choosenSong
	global lb

	songs = os.listdir("PlayList")
	size = len(songs)
	index = random.randint(0, size-1)
	
	choosenSong =  re.sub('\d+', '', songs[index])
	choosenSong = choosenSong.replace('.mp', '')
	
	print (str(index), choosenSong)
	
	mixer.music.load("PlayList/" + songs[index])
	mixer.music.play()
	
	lb.configure(text=choosenSong)
	lb.update()

def play():
	time.sleep(3)
	mixer.music.play()

def resume():
	mixer.music.unpause()

def pause():
	mixer.music.pause()

def stop():
	mixer.music.stop()

def feedback():
	os.system('python Ensemble.py')

t1 = Thread(target=feedback)
t2 = Thread(target=play)

def main():

	global root

	# img = Image.open("4a.jpg")
	# photo = tk.PhotoImage("4a.jpg")
	# background_label = Label(root, image=photo)
	# background_label.place(x=0, y=0, relwidth=1, relheight=1)
	# background_label.image = photo

	root.geometry("400x600")

	# label = Label(root, text='Genre Detection of Songs')
	# label.grid(row=9, column=6, pady=10)

	img = ImageTk.PhotoImage(Image.open("play-it-again-720x540.png"))
	panel = tkinter.Label(root, image = img,width=500,height=500)
	panel.pack(side="bottom",fill="x",expand=1)

	b0.pack(side="left",fill="x",expand=1)
	b4 = Button(root, text='Feedback', bg="white", fg='black', bd=4, relief='raised',  command=stop)
	b4.pack(side="right",fill="x",expand=1)
	b1 = Button(root,text='Resume', bg="white", fg='black', bd=4, relief='raised', command=resume)
	b1.pack(side="left")
	b3 = Button(root, text='Pause', bg="white", fg='black', bd=4, relief='raised',  command=pause)
	b3.pack(side="left")
	b2 = Button(root, text='Next Song', bg="white", fg='black', bd=4, relief='raised', command=nextSong)
	b2.pack(side="right")
	
	root.mainloop()

if __name__ == '__main__':
	main()