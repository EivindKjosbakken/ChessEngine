import random
import numpy as np
import gym
import chess

import gym_chess
import gym.spaces

from typing import List

from stockfish import Stockfish

from lib.HelperFuctions import *
stockfish = Stockfish(path=r"stockfish\stockfish-windows-x86-64-avx2.exe")
env = gym.make('ChessAlphaZero-v0')
env.reset()

def mineGames(numGames: int):
    """Mines numGames games of moves"""
    MAX_MOVES = 500  # Don't continue games after this number

    for i in range(numGames):
        currentGameMoves = []
        currentGamePositions = []
        board = chess.Board()
        stockfish.set_position([])
        for i in range(MAX_MOVES):
            #randomly choose from those 3 moves
            moves = stockfish.get_top_moves(3)
            #if less than 3 moves available, choose first one, if none available, exit
            if (len(moves) == 0):
                print("game is over")
                break
            elif (len(moves) == 1):
                move = moves[0]["Move"]
            elif (len(moves) == 2):
                move = random.choices(moves, weights=(80, 20), k=1)[0]["Move"]
            else:
                move = random.choices(moves, weights=(80, 15, 5), k=1)[0]["Move"]

            currentGamePositions.append(stockfish.get_fen_position())
            currentGameMoves.append(move) #make sure to add str version of move before changing format
            move = chess.Move.from_uci(str(move)) #convert to format chess package likes
            board.push(move)
            stockfish.set_position(currentGameMoves)
            if (checkEndCondition(board)):
                print("game is over")
                break
    saveData(currentGameMoves, currentGamePositions)



if __name__ == "__main__":
    # mineGames(3)
    testBoard = runGame(12, "movesAndPositions4.npy")
    # encodeAllMovesAndPositions()

