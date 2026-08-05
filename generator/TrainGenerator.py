
import sqlite3
import numpy as np
import boardlib
import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np





class KilterClimbGenData:
  def __init__(self, angle, grade, string, nomatch):
    self.angle = angle
    self.grade = grade
    self.string = string
    self.nomatch = nomatch

class KilterDataSet(Dataset):
    def __init__(self, climbs):
        self.climbs = climbs
    def __len__(self):
        return len(self.climbs)
    def __getitem__(self, i): #turns into tensor
        holds = self.climbs[i].string
        token = ""
        tokens = [] #this stores all the tokens where a token is in an example p123r12 123 and 12
        angleRound = round(self.climbs[i].angle / 5) * 5
        tokens.append(angleRound // 5 + 1511)
        tokens.append(self.climbs[i].grade + 70/5 + 1511)
        tokens.append(self.climbs[i].nomatch +  70/5 + 1511  + 39)
        for char in holds:  # parse the string into tokens
            if char == 'p' or char == 'r':
                if (token != ""):
                    tokens.append(int(token))
                token = ""
            else:
                token = token + char


        for x in range(153 - len(tokens)): #construct padding
            tokens.append(0)


        attention = [] #define which tokens are real vs padding
        for x in range(153):
            if tokens[x] != 0:
                attention.append(1)
            else:
                attention.append(0)

        attentionTensor = torch.tensor(attention, dtype = torch.float32)
        tokenTensor = torch.tensor(tokens, dtype = torch.int64)
        return attentionTensor, tokenTensor

class KilterGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4846, 64) #lookup table for turning our holds to vectors need 0 - 1510 so 1511 total
        self.trans1 = nn.TransformerEncoderLayer(d_model= 64, nhead = 8, dim_feedforward=256, batch_first=True)
        self.trans2 = nn.TransformerEncoderLayer(d_model= 64, nhead = 8, dim_feedforward=256, batch_first=True)
        self.trans3 = nn.TransformerEncoderLayer(d_model= 64, nhead = 8, dim_feedforward=256, batch_first=True)
        self.trans4 = nn.TransformerEncoderLayer(d_model= 64, nhead = 8, dim_feedforward=256, batch_first=True)
        #self.embedding = nn.TransformerEncoder(encoder_layer= 64, num_layers= 4) used to automatically make 4 layers with 64 parameters each
        self.linear = nn.Linear(64, 4846)

    def forward(self, tokens, attention, casual):
        attention = ~attention.bool()
        x = self.embedding(tokens)
        x = self.trans1(x, src_key_padding_mask = attention, src_mask=casual)
        x = self.trans2(x, src_key_padding_mask = attention, src_mask=casual)
        x = self.trans3(x, src_key_padding_mask = attention, src_mask=casual)
        x = self.trans4(x, src_key_padding_mask = attention, src_mask=casual)
        x = self.linear(x)
        return x


def trainGuesser():
    with open('KilterClimbsGenerationData.pkl', 'rb') as file:
        temp = pickle.load(file)
    climbs = KilterDataSet(temp)
    model = KilterGenerator()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    train_data = DataLoader(climbs, batch_size = 64, shuffle = True, num_workers = 1)

    for epoch in range(25):
        print("We are on epoch: " + str(epoch))
        for attention, token in train_data:
            input = token[:, :-1] # only let the ai see up to the token we want it to predict
            target = token[:,1:] #let it see the actual token it shouldve predicted
            target= target.to(device)
            input = input.to(device)
            target = target.reshape(-1)
            casual = torch.nn.Transformer.generate_square_subsequent_mask(152)
            attention, token, casual = attention.to(device), token.to(device), casual.to(device)
            attention = attention[:,:-1]
            prediction = model(input, attention, casual)
            prediction = prediction.reshape(-1, 4846)
            loss = error(prediction, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.save(model.state_dict(), "checkpoints/GenAI_epoch:" + str(epoch))

if __name__ == "__main__":
    trainGuesser()
