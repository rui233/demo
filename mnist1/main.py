import torch
from torchvision import transforms
from  lenet import LeNet5
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
from Mouse import Mouse
import os

device = torch.device('cpu')
transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor()
    ])


def creat_image():
    print("create image")
    mn = Mouse()
    mn.create_image()

def predict(curr_frame):
    net = LeNet5()
    net.load_state_dict(torch.load('/home/ray/git/mnist1/params.pkl'))
    net.to(device)
    torch.no_grad()

    trans_frame = transform(curr_frame).unsqueeze(0)

    input_frame = trans_frame.to(device)

    outputs = net(input_frame)
    _, predicted = torch.max(outputs, 1)

#   print("predict number is:", outputs)
    print("number is:", predicted)
    return predicted


class Application(tk.Frame):
    def __init__(self, root):
        super().__init__(root)

        root.title('digital recognition')
        win = Canvas(root, width=500, height=50)
        win.pack()
        
        Label(root, text="welcome to use digital recognition system").pack(side=TOP)
        Label(root, text="1.press m to write digital").pack(side=TOP, anchor=W) 
        Label(root, text="2.press w, to crop the region of digital").pack(side=TOP, anchor=W)
        Label(root, text="3.press q, close the draw pad").pack(side=TOP, anchor=W)
        Label(root, text="4.click fesh button,to fresh new picture").pack(side=TOP, anchor=W)
        Label(root, text="5.click recogniton button").pack(side=TOP, anchor=W) 
        
        self.numLabel = Label(root, text='', relief=RAISED, fg="black", font=("bold", 50, "bold"))
        self.numLabel.pack(side=BOTTOM, anchor=CENTER)
        Label(root, text='').pack(side=TOP, anchor=W)

        fm = Frame(root)

        Button(fm, text="create picture", command=creat_image).pack(side=TOP, anchor=W, fill=X, expand=YES)
        Button(fm, text="fresh", command=self.fresh).pack(side=TOP, anchor=W, fill=X, expand=YES)       
        Button(fm, text="digital recognition", command=self.recognition).pack(side=TOP, anchor=W, fill=X, expand=YES)
        Label(root, text="The number is:").pack(side=BOTTOM, anchor=S) 
        fm.pack(side=LEFT, fill=BOTH, expand=YES, padx=20)

        self.pilImage = Image.open("/home/ray/git/mnist1/only.png")
        self.tkImage = ImageTk.PhotoImage(image=self.pilImage)
        self.label = Label(root, image=self.tkImage)
        self.label.pack()

    
    def fresh(self):
        self.png = tk.PhotoImage(file="/home/ray/git/mnist1/only.png")
        self.label.configure(image=self.png)

    def recognition(self):
        print("begin to recogition")        
        img = Image.open('/home/ray/git/mnist1/only.png')
        img = img.resize((28, 28))
        img.show()        
        net_img = img.convert('L')
        net_img.show()
        display = predict(net_img)
        print(display.item())
        self.numLabel.configure(text=str(display.item()))
        
if __name__ == '__main__':
    root = Tk()
    app = Application(root)
    root.mainloop()
