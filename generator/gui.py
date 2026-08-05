import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pickle
from mpmath.libmp.gammazeta import f3
import torch
from TrainGenerator import KilterGenerator

# TODO: import your model class so you can load a checkpoint
# from TrainGenerator import KilterGenerator


class KilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kilter Board Generator")

        #--- load your saved data here ---
        with open("KilterMap.pkl", "rb") as f:
            self.coord_map = pickle.load(f)
            self.role_to_color = pickle.load(f)
            self.grade_to_name = pickle.load(f)

        #--- load your trained model here ---
        self.model = KilterGenerator()
        self.model.load_state_dict(torch.load("checkpoints/GenAI_epoch:24"))
        self.model.eval()

        self.root.configure(bg='black')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background = 'black')
        style.configure('TLabel', background = 'black', foreground = 'white')
        style.configure('TButton', background = 'black', foreground = 'white')
        style.configure('TEntry', fieldbackground = 'black', foreground = 'white')

        self.build_input_panel()
        self.build_canvas()

    def build_input_panel(self):
        frame = ttk.Frame(self.root)
        frame.pack(side="left", fill="y", padx=10, pady=10)

        ttk.Label(frame, text="Angle (0-70, 5 degree intervals)").pack()
        self.angle_var = tk.IntVar(value=40)
        ttk.Entry(frame, textvariable=self.angle_var).pack()

        ttk.Label(frame, text="V Grade (0-13) (13< rounded to 13)").pack()
        self.grade_var = tk.IntVar(value=10)
        ttk.Entry(frame, textvariable=self.grade_var).pack()

        ttk.Label(frame, text="No Match (0 = Match, 1 = No Match)").pack()
        self.nomatch_var = tk.IntVar(value=0)
        ttk.Entry(frame, textvariable=self.nomatch_var).pack()

        ttk.Button(frame, text="Generate", command=self.on_generate).pack(pady=20)


    def build_canvas(self):
        # scale the image down so it fits comfortably on screen (1080px tall screen)
        img = Image.open("kilterboard.jpeg")
        target_height = 950
        scale = target_height / img.height
        new_size = (int(img.width * scale), target_height)
        img = img.resize(new_size)

        self.width, self.height = new_size

        self.canvas = tk.Canvas(self.root, width=new_size[0], height=new_size[1], bg="#1a1a1a")
        self.canvas.pack(side="left")

        self.board_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.board_image)
    def generateClimbs(self, angle, grade , nomatch):
        token = []
        roundedAngle = round(angle // 5) * 5
        token.append(roundedAngle // 5 + 1511)
        token.append(grade + 70/5 + 1511)
        token.append(nomatch + 70/5 + 1511 + 39)
        with torch.no_grad():
            while True:
                tensor = torch.tensor(token, dtype = torch.int64)
                tensor = tensor.unsqueeze(0)
                casual = torch.nn.Transformer.generate_square_subsequent_mask(len(token))
                attention = torch.ones((1,len(token)), dtype = torch.float32)
                prediction = self.model(tensor, attention, casual)
                last = prediction[:,-1,:]
                prob = torch.softmax(last, dim = -1)
                nextToken = torch.multinomial(prob, num_samples = 1).item()
                token.append(nextToken)
                if nextToken == 0 or len(token) >= 150:
                    break
        return token
    def parseClimb(self):
        angle = self.angle_var.get()
        grade = self.grade_to_name["V" + str(min(13, self.grade_var.get()))]
        nomatch = self.nomatch_var.get()

        tokens = self.generateClimbs(angle, grade, nomatch)
        tokens.pop(0)
        tokens.pop(0)
        tokens.pop(0)  #remove our leftover stuff from feeding to the ai.
        if (tokens[-1] == 0):
            tokens.pop(-1)

        holds = []
        for i in range(0, len(tokens) - 1, 2):
            hold = []
            placement = tokens[i]
            role = tokens[i + 1]
            x,y = self.coord_map[placement]
            color = self.role_to_color[role]
            hold = (x,y,color)
            holds.append(hold)
        return holds


    def on_generate(self):

        holds = []
        for i in range(1000):
            tempHolds = self.parseClimb()
            green = 0
            purple = 0
            for hold in tempHolds:
                if hold[2] == 'start':
                    green = green + 1
                if hold[2] == 'finish':
                    purple = purple + 1
            if purple <= 2 and purple > 0 and green <= 2 and green > 0:
                holds = tempHolds
                break

        self.draw_climb(holds)







    def draw_climb(self, holds):

        self.canvas.delete("hold")
        for i in range(len(holds)):
            centerX = holds[i][0] * (self.width/144)
            centerY = self.height - (holds[i][1] * ( self.height/ 156))
            finalColor = holds[i][2]

            if (finalColor == 'foot'):
                finalColor = 'yellow'
            if (finalColor == 'middle'):
                finalColor = 'cyan'
            if (finalColor == 'finish'):
                finalColor = 'magenta'
            if (finalColor == 'start'):
                finalColor = 'lime'
            radius = 12
            self.canvas.create_oval(centerX - radius, centerY - radius, centerX + radius, centerY + radius,
                                     outline=finalColor, fill="", width=3, tags="hold")


if __name__ == "__main__":
    root = tk.Tk()
    app = KilterGUI(root)
    root.mainloop()
