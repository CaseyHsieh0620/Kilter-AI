import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
import numpy as np


class KilterClimb:
  def __init__(self, angle, grade, placementX, placementY, color, nomatch, matrix, ascent, benchmark):
    self.angle = angle
    self.grade = grade
    self.placementX = placementX
    self.placementY = placementY
    self.color = color
    self.nomatch = nomatch
    self.matrix = matrix
    self.ascents = ascent
    self.benchmark = benchmark
# import our kilter class

class KilterDataSet(Dataset):
    def __init__(self, climbs, grades):
        self.climbs = climbs
        self.grades = grades
    def __len__(self):
        return len(self.climbs)
    def __getitem__(self,i):   # turn the matricies into seperate matricies one for each t ype of hold and overlay them into a tensor
        climb = self.climbs[i]

        matrixStart = (climb.matrix == 'g').astype(np.float32)
        matrixHand = (climb.matrix == 'b').astype(np.float32)
        matrixEnd = (climb.matrix == 'p').astype(np.float32)
        matrixFeet = (climb.matrix == 'y').astype(np.float32)

        matrixStack = np.stack([matrixFeet, matrixHand,matrixStart,matrixEnd], axis = 0)    #convert them into tensors to feed to the ai
        positionTensor = torch.tensor(matrixStack, dtype = torch.float32)
        attributeStack = np.stack([ climb.angle, climb.nomatch], axis = 0)
        attributeTensor = torch.tensor(attributeStack, dtype = torch.float32)
        gradeTensor = torch.tensor(self.grades[round(climb.grade)], dtype = torch.long) 
        weightTensor = torch.log1p(torch.tensor(float(climb.ascents),dtype = torch.float32))
        return positionTensor, attributeTensor , gradeTensor, weightTensor

    
class KilterAi(nn.Module):
    def __init__(self, num_grades):
        super().__init__()
        self.holdQuality = nn.Parameter(torch.zeros(44,47))
        self.conv1 = nn.Conv2d(in_channels = 5, out_channels = 32, kernel_size = 3, padding = 1)  #in channels describes input, out is the output, kernal size is how big the sliding window grid is and padding is how much border you get with 1 extra boroder of zeros
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1) # convolution : https://developer.nvidia.com/discover/convolution
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1) # construct a convolution network that grows from size 4 -> 64 -> 128 -> 256 size
        self.bn4 = nn.BatchNorm2d(256) #Re normalizes so outputs dont' varry wildly
        # self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()  # turrn this into something linear can see and do math on
        self.linear1 = nn.Linear(in_features= 256*5*5 + 2, out_features = 64) # do the actual math on it (this is an actual layer)
        self.linear2 = nn.Linear(in_features = 64, out_features = num_grades)
        self.pool2 = nn.MaxPool2d(kernel_size = 2)

    def forward(self, grid, aux): # this describes our foward pass through the ai
        holdDifficulty = (grid.sum(dim = 1, keepdim = True) > 0).float()
        quality = holdDifficulty * self.holdQuality.unsqueeze(0).unsqueeze(0)
        grid = torch.cat([grid, quality], dim = 1)
        x = torch.relu(self.bn1(self.conv1(grid)))
        x = self.pool2(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn4(self.conv4(x)))
        # x = self.pool(x)
        x = self.flatten(x)
        x = torch.cat([x, aux], dim = 1)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def getGrades(climbs):   #return a list of grades
    gradeset = set()
    for climb in climbs:
        if climb.grade in gradeset:
            continue
        if climb.grade not in gradeset:
            gradeset.add(round(climb.grade))
    return sorted(gradeset)


def trainGuesser():
    with open('climbs.pkl', 'rb') as file:
        climbs = pickle.load(file)
    climbset = getGrades(climbs)
    map = {}
    index = 0
    for grade in climbset:
        map[grade] = index
        index = index + 1
    data = KilterDataSet(climbs, map)
    eval, train = torch.utils.data.random_split(data, [300, len(data) - 300], generator = torch.Generator().manual_seed(42))
    train_data = DataLoader(train, batch_size = 64, shuffle = True, num_workers = 4)
    eval_data = DataLoader(eval, batch_size = 64, shuffle = False)
    grades = len(climbset)
    model = KilterAi(grades)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    for epoch in range(50):
        print("We are on epoch: " + str(epoch))
        epochLoss = 0
        batchCount = 0
        for grids,auxs,grades, weight in train_data:
            grids, auxs, grades = grids.to(device), auxs.to(device), grades.to(device)
            prediction = model(grids, auxs) #forward pass
            loss = error(prediction, grades) # loss function
            loss.backward()  #figure out gradient backwards
            optimizer.step()   # step in the right directions
            optimizer.zero_grad()  #zero out the gradient for the next batch
            epochLoss = epochLoss + loss.item()
            batchCount = batchCount + 1
        print("finished epoch: " + str(epoch) + " the loss was: " + str(epochLoss/batchCount))
        torch.save(model.state_dict(), "kilterAI")
        total = 0
        meanAverageError = 0;
        bucketStats = {}
        for matrix, aux, grades, weight in eval_data:
            matrix, aux, grades = matrix.to(device), aux.to(device), grades.to(device)
            model.eval()
            with torch.no_grad():
                pred = model(matrix,aux )
                predicted_index = torch.argmax(pred, dim = 1)
                evalError = torch.abs(predicted_index - grades)
                meanAverageError = meanAverageError + evalError.sum().item()
                total = total + grades.size(0)
                for i in range(grades.size(0)):
                    trueGrade = climbset[grades[i].item()]
                    bucket = (trueGrade // 5) * 5
                    if bucket not in bucketStats:
                        bucketStats[bucket] = [0, 0]
                    bucketStats[bucket][0] += evalError[i].item()
                    bucketStats[bucket][1] += 1

        print("The MAE is: " + str(meanAverageError/total))
        for bucket in sorted(bucketStats):
            totalError, count = bucketStats[bucket]
            print("grades " + str(bucket) + "-" + str(bucket+4) + ": MAE " + str(totalError/count) + " over " + str(count) + " climbs")


if __name__ == "__main__":
    trainGuesser()
