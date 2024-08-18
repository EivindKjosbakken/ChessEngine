import concurrent.futures
import os
import random
import tkinter as tk
from tkinter import messagebox

import gym
from stockfish import Stockfish

from libs.Training import *
stockfish = Stockfish(path=r"../stockfish/stockfish-windows-x86-64-avx2.exe")
env = gym.make('ChessAlphaZero-v0')
env.reset()

def mineGames(numGames: int, MAX_MOVES: int = 500) -> None:
    # Using ProcessPoolExecutor with a limited number of workers
    max_workers = 16  # Adjust based on your CPU capacity
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of arguments using list comprehension
        data_list = [MAX_MOVES] * numGames

        # Submit tasks to the executor
        futures = [executor.submit(mineOneGame, data) for data in data_list]

        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will re-raise any exceptions caught during the task execution
            except Exception as exc:
                print(f'Task generated an exception: {exc}')


def mineOneGame(MAX_MOVES):
    currentGameMoves = []
    currentGamePositions = []
    board = chess.Board()
    stockfish.set_position([])
    for i in range(MAX_MOVES):
        # randomly choose from those 3 moves
        moves = stockfish.get_top_moves(3)
        # if less than 3 moves available, choose first one, if none available, exit
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
        currentGameMoves.append(move)  # make sure to add str version of move before changing format
        move = chess.Move.from_uci(str(move))  # convert to format chess package likes
        board.push(move)
        stockfish.set_position(currentGameMoves)
        if (checkEndCondition(board)):
            print("game is over")
            saveData(currentGameMoves, currentGamePositions)
            break


def play_games():
    games = int(amount_of_games_entry.get())
    # Function to play games here
    messagebox.showinfo("Info", f"Playing {games} games.")

def show_total_games():
    raw_data_dir = "../data/rawdata"  # Replace this with the path to your raw data directory
    try:
        total_games = len(os.listdir(raw_data_dir))-1
        messagebox.showinfo("Info", f"Total games played: {total_games}")
    except FileNotFoundError:
        messagebox.showerror("Error", "Raw data directory not found.")

def show_positions():
    # Function to show played positions here
    messagebox.showinfo("Info", "Played positions: <insert played positions here>")

def show_moves():
    # Function to show played moves here
    messagebox.showinfo("Info", "Played moves: <insert played moves here>")

def create_main_screen():
    main_screen = tk.Tk()
    main_screen.title("Chess Program")
    
    menu = tk.Menu(main_screen)
    main_screen.config(menu=menu)
    
    games_menu = tk.Menu(menu)
    menu.add_cascade(label="Menu", menu=games_menu)
    games_menu.add_command(label="Play games", command=play_games)
    games_menu.add_command(label="Show total games played", command=show_total_games)
    games_menu.add_command(label="Show played positions", command=show_positions)
    games_menu.add_command(label="Show played moves", command=show_moves)
    
    amount_of_games_label = tk.Label(main_screen, text="Amount of games:")
    amount_of_games_label.pack()
    global amount_of_games_entry
    amount_of_games_entry = tk.Entry(main_screen)
    amount_of_games_entry.pack()
    
    main_screen.mainloop()

if __name__ == "__main__":
    # create_main_screen()
    user_input = input("Please enter an integer: ")

    try:
        # Convert the input to an integer
        user_integer = int(user_input)
        print("You entered:", user_integer)
        mineGames(user_integer)
        # Ask the user if they want to proceed with training
        train_input = input("Do you want to proceed with training? (yes/no): ").lower()

        if train_input == "yes":
            encodeAllMovesAndPositions()
            runTraining()
        elif train_input == "no":
            print("Training will not proceed.")
        else:
            print("Invalid input for training. Please enter 'yes' or 'no'.")
    except ValueError as e:
     print("Invalid input. Please enter a valid integer.")
     print("An error occurred:", str(e))
    
    

