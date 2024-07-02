import glob
from datetime import datetime
from pathlib import Path

import chess
import numpy as np
import torch
from libs.HelperFunctions import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def LoadingTrainingData(FRACTION_OF_DATA = 1, BATCH_SIZE = 32):
    #loading training data

    allMoves = []
    allBoards = []

    files = (glob.glob(r"../data/preparedData/moves*.npy"))

    for f in files:
        currUuid = f.split("moves")[-1].split(".npy")[0]
        try:
            moves = np.load(f"../data/preparedData/moves{currUuid}.npy", allow_pickle=True)
            boards = np.load(f"../data/preparedData/positions{currUuid}.npy", allow_pickle=True)
            if (len(moves) != len(boards)):
                print("ERROR ON i = ", currUuid, len(moves), len(boards))
            allMoves.extend(moves)
            allBoards.extend(boards)
        except:
            print("error: could not load ", currUuid, ", but is still going")
            pass
            

    allMoves = np.array(allMoves)[:(int(len(allMoves) * FRACTION_OF_DATA))]
    allBoards = np.array(allBoards)[:(int(len(allBoards) * FRACTION_OF_DATA))]
    assert len(allMoves) == len(allBoards), "MUST BE OF SAME LENGTH"


    #flatten out boards
    # allBoards = allBoards.reshape(allBoards.shape[0], -1)

    trainDataIdx = int(len(allMoves) * 0.8)

    #NOTE transfer all data to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    allBoards = torch.from_numpy(np.asarray(allBoards)).to(device)
    allMoves = torch.from_numpy(np.asarray(allMoves)).to(device)

    training_set = torch.utils.data.TensorDataset(allBoards[:trainDataIdx], allMoves[:trainDataIdx])
    test_set = torch.utils.data.TensorDataset(allBoards[trainDataIdx:], allMoves[trainDataIdx:])
    # Create data loaders for our datasets; shuffle for training, not for validation

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"loaded {len(allMoves)} moves and positions")
    return [training_loader, validation_loader]

#model
class Model(torch.nn.Module):
    

    def __init__(self):
        super(Model, self).__init__()
        self.INPUT_SIZE = 896 
        # self.INPUT_SIZE = 7*7*13 #NOTE changing input size for using cnns
        self.OUTPUT_SIZE = 4672 # = number of unique moves (action space)
        
		#can try to add CNN and pooling here (calculations taking into account spacial features)

        #input shape for sample is (8,8,14), flattened to 1d array of size 896
        # self.cnn1 = nn.Conv3d(4,4,(2,2,4), padding=(0,0,1))

        self.activation = torch.nn.Tanh()   
        # self.activation = torch.nn.ReLU()

        self.linear1 = torch.nn.Linear(self.INPUT_SIZE, 1000)
        self.linear2 = torch.nn.Linear(1000, 1000)
        self.linear3 = torch.nn.Linear(1000, 1000)
        self.linear4 = torch.nn.Linear(1000, 200)
        self.linear5 = torch.nn.Linear(200, self.OUTPUT_SIZE)
        self.softmax = torch.nn.Softmax(1) #use softmax as prob for each move, dim 1 as dim 0 is the batch dimension
 
    def forward(self, x): #x.shape = (batch size, 896)
        x = x.to(torch.float32)
        # x = self.cnn1(x) #for using cnns
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = self.activation(x)
        x = self.linear5(x)
        # x = self.softmax(x) #do not use softmax since you are using cross entropy loss
        return x

    def predict(self, board : chess.Board):
        """takes in a chess board and returns a chess.move object. NOTE: this function should definitely be written better, but it works for now"""
        with torch.no_grad():
            encodedBoard = encodeBoard(board)
            encodedBoard = encodedBoard.reshape(1, -1)
            encodedBoard = torch.from_numpy(encodedBoard)
            res = self.forward(encodedBoard)
            probs = self.softmax(res)

            probs = probs.numpy()[0] #do not want tensor anymore, 0 since it is a 2d array with 1 row

            #verify that move is legal and can be decoded before returning
            while len(probs) > 0: #try max 100 times, if not throw an error
                moveIdx = probs.argmax()
                try: #TODO should not have try here, but was a bug with idx 499 if it is black to move
                    uciMove = decodeMove(moveIdx, board)
                    if (uciMove is None): #could not decode
                        probs = np.delete(probs, moveIdx)
                        continue
                    move = chess.Move.from_uci(str(uciMove))
                    if (move in board.legal_moves): #if legal, return, else: loop continues after deleting the move
                        return move 
                except:
                    pass
                probs = np.delete(probs, moveIdx) #TODO probably better way to do this, but it is not too time critical as it is only for predictions
                                             #remove the move so its not chosen again next iteration
            
            #return random move if model failed to find move
            moves = board.legal_moves
            if (len(moves) > 0):
                print(f"Returning one of {len(moves)} moves")
                return np.random.choice(list(moves))
            print("Your predict function could not find any legal/decodable moves")
            return None #if no legal moves found, return None
            # raise Exception("Your predict function could not find any legal/decodable moves")
        
#helper functions for training
def train_one_epoch(model, optimizer, loss_fn, epoch_index, tb_writer, training_loader):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):

        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            # print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def createBestModelFile():
    #first find best model if it exists:
    path = Path('../data/savedModels/bestModel.txt')

    if not (path.is_file()):
        #create the files
        f = open(path, "w")
        f.write("10000000") #set to high number so it is overwritten with better loss
        f.write("\ntestPath")
        f.close()

def saveBestModel(vloss, pathToBestModel, epoch_number):
    f = open("../data/savedModels/bestModel.txt", "w")
    f.write(str(vloss.item()))
    f.write("\n")
    f.write(pathToBestModel)
    print("NEW BEST MODEL FOUND WITH LOSS:", vloss)


def retrieveBestModelInfo():
    f = open('../data/savedModels/bestModel.txt', "r")
    bestLoss = float(f.readline())
    bestModelPath = f.readline()
    f.close()
    return bestLoss, bestModelPath
#hyperparams
EPOCHS = 500
LEARNING_RATE = 0.001
MOMENTUM = 0.9

def runTraining():

    createBestModelFile()

    bestLoss, bestModelPath = retrieveBestModelInfo()
    trainDataLoader, testDataLoader = LoadingTrainingData()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    model = Model()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_vloss = 1_000_000.

    for epoch in tqdm(range(EPOCHS)):
        if (epoch_number % 5 == 0):
            print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, optimizer, loss_fn, epoch_number, writer, trainDataLoader)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.

        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(trainDataLoader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)

                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        #only print every 5 epochs
        if epoch_number % 5 == 0:
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

            if (bestLoss > best_vloss): #if better than previous best loss from all models created, save it
                model_path = '../data/savedModels/model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)
                saveBestModel(best_vloss, model_path, epoch_number)

        epoch_number += 1

    print("\n\nBEST VALIDATION LOSS FOR ALL MODELS: ", bestLoss)