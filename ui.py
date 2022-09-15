import os
import tkinter as tk
from dotenv import load_dotenv

load_dotenv()

from ravpy.distributed.participate import participate
from ravpy.initialize import initialize


def signin():
    token = entryToken.get()
    if token == "":
        label3 = tk.Label(root, text='Enter token to signin', fg='red', font=('Arial', 14))
        canvas1.create_window(250, 375, window=label3)
    else:
        entryToken.pack_forget()
        label1.pack_forget()
        label2.pack_forget()

        label3 = tk.Label(root, text='Log\n', fg='black', font=('Arial', 14))
        label3.pack()
        # canvas1.create_window(250, 375, window=label3)
        label3['text'] += "Initializing...\n"
        initialize(token)
        label3['text'] += "Initialized\n"
        participate()
        label3['text'] += "Participated\n"



class RavpyUI():
    def __init__(self):
        pass


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Ravpy')
    canvas1 = tk.Canvas(root, width=500, height=500)
    canvas1.pack()

    label1 = tk.Label(root, text='Sign in', fg='black', font=('helvetica', 16, 'bold'))
    canvas1.create_window(250, 150, window=label1)

    label2 = tk.Label(root, text='Enter Token:', fg='black', font=('helvetica', 14))
    canvas1.create_window(250, 200, window=label2)

    tokenString = tk.StringVar()
    entryToken = tk.Entry(root, width=30, textvariable=tokenString, font=("helvetica", 20))
    entryToken.pack(padx=10, pady=10)

    canvas1.create_window(250, 250, window=entryToken)

    button1 = tk.Button(text='Submit', command=signin, bg='brown', fg='black')
    canvas1.create_window(250, 300, window=button1)

    root.mainloop()
