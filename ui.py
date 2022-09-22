from dotenv import load_dotenv

load_dotenv()

import tkinter as tk
from ravpy.utils import verify_token
from ravpy.distributed.participate import participate
from ravpy.initialize import initialize


class RavpyUI(object):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Ravpy')

        self.signin_canvas = None
        self.entry_token = None
        self.dashboard_canvas = None

        self.create_signin_canvas()

    def create_signin_canvas(self):
        self.signin_canvas = tk.Canvas(self.root, width=500, height=500)
        self.signin_canvas.pack()

        signin_label = tk.Label(self.root, text='Sign in', fg='black', font=('helvetica', 16, 'bold'))
        enter_token_label = tk.Label(self.root, text='Enter Token:', fg='black', font=('helvetica', 14))

        token_string = tk.StringVar()
        self.entry_token = tk.Entry(self.root, width=30, textvariable=token_string, font=("helvetica", 20))
        self.entry_token.pack(padx=10, pady=10)

        signin_button = tk.Button(text='Submit', command=self.signin, bg='brown', fg='black')

        self.signin_canvas.create_window(250, 150, window=signin_label)
        self.signin_canvas.create_window(250, 200, window=enter_token_label)
        self.signin_canvas.create_window(250, 250, window=self.entry_token)
        self.signin_canvas.create_window(250, 300, window=signin_button)

    def create_dashboard_canvas(self, token):
        self.dashboard_canvas = tk.Canvas(self.root, width=500, height=500)
        self.dashboard_canvas.pack()

        label3 = tk.Label(self.root, text='Log\n', fg='black', font=('Arial', 14))
        self.dashboard_canvas.create_window(250, 150, window=label3)

        # canvas1.create_window(250, 375, window=label3)
        label3['text'] += "Initializing...\n"
        initialize(token)
        label3['text'] += "Initialized\n"
        participate()
        label3['text'] += "Participated\n"

    def start(self):
        self.root.mainloop()

    def signin(self):
        token = self.entry_token.get()
        if token == "":
            label3 = tk.Label(self.root, text='Enter token to signin', fg='red', font=('Arial', 14))
            self.signin_canvas.create_window(250, 375, window=label3)
        else:
            if verify_token(token=token):
                self.signin_canvas.pack_forget()
                self.create_dashboard_canvas(token)


if __name__ == '__main__':
    ui = RavpyUI()
    ui.start()
