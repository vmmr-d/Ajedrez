import io
import os
import chess
import chess.pgn
import pandas as pd
import numpy as np  # Importación necesaria para fen_to_matrix
from tqdm import tqdm  # Importación necesaria para el progreso en move_validos

# Definimos una función para convertir el contenido PGN a un DataFrame            
def pgn_to_dataframe(pgn):
    games = []
    pgn_io = io.StringIO(pgn)

    while True:
        try:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break

            headers = game.headers

            game_data = {
                "Event": headers.get("Event", ""),
                "Result": headers.get("Result", ""),
                "WhiteElo": headers.get("WhiteElo", "0"),
                "BlackElo": headers.get("BlackElo", "0"),
                "TimeControl": headers.get("TimeControl", ""),
                "ECO": headers.get("ECO", ""),
                "Opening": headers.get("Opening", "")}
            
            try:
                game_data["WhiteElo"] = int(game_data["WhiteElo"]) if game_data["WhiteElo"].isdigit() else 0
                game_data["BlackElo"] = int(game_data["BlackElo"]) if game_data["BlackElo"].isdigit() else 0

            except ValueError:
                game_data["WhiteElo"] = 0
                game_data["BlackElo"] = 0    

            try:
                result_mapping = {'1-0': 0, '0-1': 1, '1/2-1/2': 2}
                game_data['Result'] = result_mapping.get(game_data['Result'], -1)  # -1 para resultados desconocidos
            except KeyError:
                game_data['Result'] = -1

            moves = []
            board = game.board()
            for move in game.mainline_moves():
                moves.append(board.san(move))
                board.push(move)
            
            game_data['Moves'] = " ".join(moves) if moves else ""
            games.append(game_data)

        except Exception as e:
            print(f"Error procesando juego PGN: {e}")
            continue

    Juegos = pd.DataFrame(games) if games else pd.DataFrame()

    # Guardar el DataFrame en formato CSV
    if not Juegos.empty:
        Juegos.to_csv('../data/raw/games.csv', index=False)
        print("Archivo guardado exitosamente.")
    else:
        print("No se generó ningún dato para guardar.")

    return Juegos

# Definimos una función para extraer el control de tiempo de un juego
def extraer_time_control(tc_str):
    try:
        if '+' in str(tc_str):
            base, increment = tc_str.split('+')
            return int(base) + int(increment)
        else:
            return int(tc_str) if str(tc_str).isdigit() else 600
    except ValueError:
        return 600

# Función para procesar los movimientos de las partidas
def moves_to_int64(moves_list):
    board = chess.Board()
    encoded_moves = []
    
    for move in moves_list:
        # Convertir el movimiento SAN a un objeto chess.Move
        chess_move = board.parse_san(move)
        
        # Codificar el movimiento como un entero único
        move_int = chess_move.from_square * 64 + chess_move.to_square
        encoded_moves.append(move_int)
        
        # Aplicar el movimiento al tablero
        board.push(chess_move)
    
    return encoded_moves

