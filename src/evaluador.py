import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chess
import chess.pgn
import io
import os
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


class Evaluador:
    """
    Modelo para extraer características de partidas de ajedrez y calcular labels de rendimiento.
    """
    
    def __init__(self):
        """
        Inicializa el extractor de características.
        """
        pass
    
    def extract_features_from_game(self, moves_str):
        """
        Extrae características de una partida a partir de una cadena de movimientos.
        
        Args:
            moves_str: String con los movimientos de la partida en formato algebraico.
        
        Returns:
            Un diccionario con características calculadas.
        """
        # Crear un tablero nuevo
        board = chess.Board()
        
        # Inicializar características
        features = {
            # Material
            'white_pawns_avg': 0,
            'black_pawns_avg': 0,
            'white_knights_avg': 0,
            'black_knights_avg': 0,
            'white_bishops_avg': 0,
            'black_bishops_avg': 0,
            'white_rooks_avg': 0,
            'black_rooks_avg': 0,
            'white_queens_avg': 0,
            'black_queens_avg': 0,
            
            # Control del centro
            'white_center_control_avg': 0,
            'black_center_control_avg': 0,
            
            # Movilidad
            'white_mobility_avg': 0,
            'black_mobility_avg': 0,
            
            # Seguridad del rey
            'white_king_safety_avg': 0,
            'black_king_safety_avg': 0,
            
            # Estructura de peones
            'white_doubled_pawns_avg': 0,
            'black_doubled_pawns_avg': 0,
            'white_isolated_pawns_avg': 0,
            'black_isolated_pawns_avg': 0,
            
            # Desarrollo
            'white_development_avg': 0,
            'black_development_avg': 0,
            
            # Capturas
            'white_captures': 0,
            'black_captures': 0,
            
            # Jaques
            'white_checks': 0,
            'black_checks': 0,
            
            # Tiempo de desarrollo (en movimientos)
            'white_development_time': 0,
            'black_development_time': 0,
            
            # Enroques
            'white_castled': 0,
            'black_castled': 0,
            
            # Longitud de la partida
            'total_moves': 0,
            
            # Movimientos en el centro
            'white_center_moves': 0,
            'black_center_moves': 0,
            
            # Ataques a piezas
            'white_attacks_avg': 0,
            'black_attacks_avg': 0,
            
            # Piezas en casillas débiles
            'white_weak_squares_avg': 0,
            'black_weak_squares_avg': 0,
            
            # Actividad de piezas
            'white_piece_activity_avg': 0,
            'black_piece_activity_avg': 0
        }
        
        # Contadores para promedios
        position_count = 0
        
        # Posiciones iniciales de piezas menores
        initial_minor_squares = [
            chess.B1, chess.G1, chess.C1, chess.F1,  # Caballos y alfiles blancos
            chess.B8, chess.G8, chess.C8, chess.F8   # Caballos y alfiles negros
        ]
        
        # Casillas centrales
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        
        # Casillas débiles típicas
        weak_squares_white = [chess.C3, chess.F3, chess.C4, chess.F4]
        weak_squares_black = [chess.C6, chess.F6, chess.C5, chess.F5]
        
        # Dividir la cadena de movimientos
        moves = moves_str.split()
        
        # Variables para seguimiento de desarrollo
        white_pieces_developed = set()
        black_pieces_developed = set()
        white_castled = False
        black_castled = False
        
        try:
            # Procesar cada movimiento
            for move_idx, move_str in enumerate(moves):
                try:
                    # Intentar hacer el movimiento
                    move = board.parse_san(move_str)
                    
                    # Comprobar capturas
                    if board.is_capture(move):
                        if board.turn == chess.WHITE:
                            features['white_captures'] += 1
                        else:
                            features['black_captures'] += 1
                    
                    # Comprobar movimientos al centro
                    to_square = move.to_square
                    if to_square in center_squares:
                        if board.turn == chess.WHITE:
                            features['white_center_moves'] += 1
                        else:
                            features['black_center_moves'] += 1
                    
                    # Realizar el movimiento
                    board.push(move)
                    
                    # Comprobar jaques
                    if board.is_check():
                        if board.turn == chess.WHITE:  # El turno ha cambiado, así que esto es un jaque de las negras
                            features['black_checks'] += 1
                        else:
                            features['white_checks'] += 1
                    
                    # Comprobar enroque
                    piece_moved = board.piece_at(to_square)
                    from_square = move.from_square
                    
                    if piece_moved and piece_moved.piece_type == chess.KING:
                        # Detectar enroque por el movimiento del rey (2 casillas)
                        if abs(chess.square_file(from_square) - chess.square_file(to_square)) > 1:
                            if piece_moved.color == chess.WHITE and not white_castled:
                                white_castled = True
                                features['white_castled'] = 1
                            elif piece_moved.color == chess.BLACK and not black_castled:
                                black_castled = True
                                features['black_castled'] = 1
                    
                    # Seguimiento de desarrollo de piezas
                    if piece_moved:
                        piece_type = piece_moved.piece_type
                        color = piece_moved.color
                        
                        # Seguimiento de desarrollo de piezas menores
                        if piece_type in [chess.KNIGHT, chess.BISHOP]:
                            piece_id = (piece_type, from_square)
                            
                            if color == chess.WHITE and from_square in [chess.B1, chess.G1, chess.C1, chess.F1]:
                                white_pieces_developed.add(piece_id)
                                if len(white_pieces_developed) == 4 and features['white_development_time'] == 0:
                                    features['white_development_time'] = move_idx // 2 + 1
                            
                            elif color == chess.BLACK and from_square in [chess.B8, chess.G8, chess.C8, chess.F8]:
                                black_pieces_developed.add(piece_id)
                                if len(black_pieces_developed) == 4 and features['black_development_time'] == 0:
                                    features['black_development_time'] = move_idx // 2 + 1
                    
                    # Analizar posición cada 4 movimientos para reducir cálculos
                    if move_idx % 4 == 0:
                        position_count += 1
                        
                        # Material
                        features['white_pawns_avg'] += len(board.pieces(chess.PAWN, chess.WHITE))
                        features['black_pawns_avg'] += len(board.pieces(chess.PAWN, chess.BLACK))
                        features['white_knights_avg'] += len(board.pieces(chess.KNIGHT, chess.WHITE))
                        features['black_knights_avg'] += len(board.pieces(chess.KNIGHT, chess.BLACK))
                        features['white_bishops_avg'] += len(board.pieces(chess.BISHOP, chess.WHITE))
                        features['black_bishops_avg'] += len(board.pieces(chess.BISHOP, chess.BLACK))
                        features['white_rooks_avg'] += len(board.pieces(chess.ROOK, chess.WHITE))
                        features['black_rooks_avg'] += len(board.pieces(chess.ROOK, chess.BLACK))
                        features['white_queens_avg'] += len(board.pieces(chess.QUEEN, chess.WHITE))
                        features['black_queens_avg'] += len(board.pieces(chess.QUEEN, chess.BLACK))
                        
                        # Control del centro
                        white_center_control = sum(len(board.attackers(chess.WHITE, sq)) for sq in center_squares)
                        black_center_control = sum(len(board.attackers(chess.BLACK, sq)) for sq in center_squares)
                        features['white_center_control_avg'] += white_center_control
                        features['black_center_control_avg'] += black_center_control
                        
                        # Movilidad
                        original_turn = board.turn
                        
                        board.turn = chess.WHITE
                        white_mobility = board.legal_moves.count()
                        features['white_mobility_avg'] += white_mobility
                        
                        board.turn = chess.BLACK
                        black_mobility = board.legal_moves.count()
                        features['black_mobility_avg'] += black_mobility
                        
                        board.turn = original_turn
                        
                        # Seguridad del rey
                        white_king_sq = board.king(chess.WHITE)
                        black_king_sq = board.king(chess.BLACK)
                        
                        if white_king_sq is not None:
                            white_king_attackers = len(board.attackers(chess.BLACK, white_king_sq))
                            white_king_defenders = len(board.attackers(chess.WHITE, white_king_sq))
                            features['white_king_safety_avg'] += white_king_defenders - white_king_attackers
                        
                        if black_king_sq is not None:
                            black_king_attackers = len(board.attackers(chess.WHITE, black_king_sq))
                            black_king_defenders = len(board.attackers(chess.BLACK, black_king_sq))
                            features['black_king_safety_avg'] += black_king_defenders - black_king_attackers
                        
                        # Estructura de peones
                        white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
                        black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
                        
                        # Peones doblados
                        white_pawn_files = [chess.square_file(sq) for sq in white_pawns]
                        black_pawn_files = [chess.square_file(sq) for sq in black_pawns]
                        
                        white_doubled_pawns = len(white_pawn_files) - len(set(white_pawn_files))
                        black_doubled_pawns = len(black_pawn_files) - len(set(black_pawn_files))
                        
                        features['white_doubled_pawns_avg'] += white_doubled_pawns
                        features['black_doubled_pawns_avg'] += black_doubled_pawns
                        
                        # Peones aislados
                        white_isolated_pawns = 0
                        black_isolated_pawns = 0
                        
                        for file in range(8):
                            # Comprobar si hay peones en esta columna
                            white_pawns_in_file = file in white_pawn_files
                            black_pawns_in_file = file in black_pawn_files
                            
                            # Comprobar si hay peones en columnas adyacentes
                            white_pawns_adjacent = (file-1 in white_pawn_files) or (file+1 in white_pawn_files)
                            black_pawns_adjacent = (file-1 in black_pawn_files) or (file+1 in black_pawn_files)
                            
                            # Contar peones aislados
                            if white_pawns_in_file and not white_pawns_adjacent:
                                white_isolated_pawns += 1
                            if black_pawns_in_file and not black_pawns_adjacent:
                                black_isolated_pawns += 1
                        
                        features['white_isolated_pawns_avg'] += white_isolated_pawns
                        features['black_isolated_pawns_avg'] += black_isolated_pawns
                        
                        # Desarrollo
                        features['white_development_avg'] += len(white_pieces_developed)
                        features['black_development_avg'] += len(black_pieces_developed)
                        
                        # Ataques a piezas
                        white_attacks = 0
                        black_attacks = 0
                        
                        for sq in chess.SQUARES:
                            piece = board.piece_at(sq)
                            if piece:
                                if piece.color == chess.WHITE:
                                    black_attacks += len(board.attackers(chess.BLACK, sq))
                                else:
                                    white_attacks += len(board.attackers(chess.WHITE, sq))
                        
                        features['white_attacks_avg'] += white_attacks
                        features['black_attacks_avg'] += black_attacks
                        
                        # Piezas en casillas débiles
                        white_weak = 0
                        black_weak = 0
                        
                        for sq in weak_squares_white:
                            piece = board.piece_at(sq)
                            if piece and piece.color == chess.WHITE and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                                white_weak += 1
                        
                        for sq in weak_squares_black:
                            piece = board.piece_at(sq)
                            if piece and piece.color == chess.BLACK and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                                black_weak += 1
                        
                        features['white_weak_squares_avg'] += white_weak
                        features['black_weak_squares_avg'] += black_weak
                        
                        # Actividad de piezas (basada en movilidad y posición)
                        white_activity = 0
                        black_activity = 0
                        
                        # Actividad basada en piezas en el centro y casillas avanzadas
                        for sq in chess.SQUARES:
                            piece = board.piece_at(sq)
                            if piece:
                                rank = chess.square_rank(sq)
                                
                                if piece.color == chess.WHITE:
                                    # Para blancas, las filas avanzadas son mejores
                                    if piece.piece_type != chess.KING:
                                        white_activity += rank * 0.1
                                    
                                    # Bonus por piezas en el centro
                                    if sq in center_squares:
                                        white_activity += 0.5
                                else:
                                    # Para negras, las filas bajas son mejores
                                    if piece.piece_type != chess.KING:
                                        black_activity += (7 - rank) * 0.1
                                    
                                    # Bonus por piezas en el centro
                                    if sq in center_squares:
                                        black_activity += 0.5
                        
                        features['white_piece_activity_avg'] += white_activity
                        features['black_piece_activity_avg'] += black_activity
                    
                    features['total_moves'] += 1

                except Exception as e:
                    # Ignorar errores en movimientos individuales
                    continue
            
            # Calcular promedios
            if position_count > 0:
                for key in features:
                    if key.endswith('_avg'):
                        features[key] /= position_count
            
            # Establecer longitud de la partida
            features['total_moves'] = len(moves) // 2  # Dividir por 2 para obtener el número de movimientos completos
            
            # Si no se completó el desarrollo, usar el número total de movimientos
            if features['white_development_time'] == 0 and len(white_pieces_developed) > 0:
                features['white_development_time'] = features['total_moves']
            if features['black_development_time'] == 0 and len(black_pieces_developed) > 0:
                features['black_development_time'] = features['total_moves']
            
        except Exception as e:
            print(f"Error al procesar partida: {e}")
        
        # Devolver diccionario de características y lista de nombres de características
        return features, list(features.keys())
 
    def calculate_performance_labels(self, df, moves_column='moves', n_jobs=-1, batch_size=1000):
        """
        Calcula etiquetas de rendimiento basadas en características de partidas de ajedrez
        y devuelve un DataFrame con todas las características y etiquetas.
        
        Args:
            df: DataFrame con datos de partidas de ajedrez
            moves_column: Nombre de la columna que contiene los movimientos
            n_jobs: Número de trabajos paralelos (-1 para usar todos los núcleos)
            batch_size: Tamaño del lote para procesamiento
            
        Returns:
            DataFrame con las características extraídas y etiquetas de rendimiento
        """
        # Inicializar listas para almacenar características y nombres de columnas
        all_features = []
        column_names = None
        
        def extract_features_wrapper(moves_str):
            features_dict, feature_names = self.extract_features_from_game(moves_str)
            if features_dict:
                feature_values = list(features_dict.values())
                return feature_values, feature_names
            else:
                return None, None
        
        # Procesar en lotes para mostrar progreso
        for start_idx in tqdm(range(0, len(df), batch_size), desc="Procesando lotes"):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            batch_features = []
            
            if n_jobs != 1:
                # Implementación paralela (requiere joblib)
                try:
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(extract_features_wrapper)(moves_str)
                        for moves_str in batch_df[moves_column]
                    )
                    
                    for features_values, feature_names in results:
                        if features_values is not None:
                            if column_names is None:
                                column_names = feature_names
                            batch_features.append(features_values)
                        else:
                            # Manejar el caso donde la extracción de características falla
                            batch_features.append(np.zeros(len(column_names)) if column_names else None)
                
                except ImportError:
                    print("Joblib no está instalado. Usando procesamiento secuencial.")
                    n_jobs = 1
            
            if n_jobs == 1:
                # Implementación secuencial
                for moves_str in batch_df[moves_column]:
                    features_values, feature_names = extract_features_wrapper(moves_str)
                    if features_values is not None:
                        if column_names is None:
                            column_names = feature_names
                        batch_features.append(features_values)
                    else:
                        # Manejar el caso donde la extracción de características falla
                        batch_features.append(np.zeros(len(column_names)) if column_names else None)
            
            # Extender la lista con las características del lote actual
            all_features.extend(batch_features)
        
        # Convertir la lista de características a un array de NumPy
        all_features_array = np.array(all_features)
        
        # Crear DataFrame con las características extraídas
        features_df = pd.DataFrame(all_features_array, columns=column_names)
        
        # Extraer características relevantes del dataframe para calcular rendimiento
        X = np.array([
            features_df['white_pawns_avg'].values,
            features_df['black_pawns_avg'].values,
            features_df['white_piece_activity_avg'].values,
            features_df['black_piece_activity_avg'].values,
            features_df['white_center_moves'].values,
            features_df['black_center_moves'].values,
            features_df['white_attacks_avg'].values,
            features_df['black_attacks_avg'].values,
            features_df['white_weak_squares_avg'].values,
            features_df['black_weak_squares_avg'].values
        ]).T  # Transponer para tener filas como partidas
        
        # Extraer resultados (0 para victoria blancas, 1 para victoria negras, 0.5 para tablas)
        # Asumiendo que la columna 'Result' contiene estos valores
        y_result = df['Result'].values
        
        # Cantidad de partidas
        n_games = len(df)
        
        # Inicializar arrays de rendimiento
        white_performance = np.zeros(n_games)
        black_performance = np.zeros(n_games)
        
        # Índices para diferentes aspectos del juego
        material_indices = [0, 1]  # white_pawns_avg, black_pawns_avg
        activity_indices = [2, 3]  # white_piece_activity_avg, black_piece_activity_avg
        center_indices = [4, 5]    # white_center_moves, black_center_moves
        attack_indices = [6, 7]    # white_attacks_avg, black_attacks_avg
        defense_indices = [8, 9]   # white_weak_squares_avg, black_weak_squares_avg
        
        # Pesos para diferentes aspectos
        weights = {
            'material': 0.3,
            'activity': 0.2,
            'center': 0.2,
            'attack': 0.2,
            'defense': 0.1
        }
        
        # Calcular rendimiento para cada partida
        for i in range(n_games):
            # Material
            white_material = X[i, material_indices[0]]
            black_material = X[i, material_indices[1]]
            total_material = white_material + black_material
            
            if total_material > 0:
                material_score_white = white_material / total_material
                material_score_black = black_material / total_material
            else:
                material_score_white = material_score_black = 0.5
            
            # Actividad de piezas
            white_activity = X[i, activity_indices[0]]
            black_activity = X[i, activity_indices[1]]
            total_activity = white_activity + black_activity
            
            if total_activity > 0:
                activity_score_white = white_activity / total_activity
                activity_score_black = black_activity / total_activity
            else:
                activity_score_white = activity_score_black = 0.5
            
            # Control del centro
            white_center = X[i, center_indices[0]]
            black_center = X[i, center_indices[1]]
            total_center = white_center + black_center
            
            if total_center > 0:
                center_score_white = white_center / total_center
                center_score_black = black_center / total_center
            else:
                center_score_white = center_score_black = 0.5
            
            # Ataques
            white_attack = X[i, attack_indices[0]]
            black_attack = X[i, attack_indices[1]]
            total_attack = white_attack + black_attack
            
            if total_attack > 0:
                attack_score_white = white_attack / total_attack
                attack_score_black = black_attack / total_attack
            else:
                attack_score_white = attack_score_black = 0.5
            
            # Defensa (inversamente proporcional a casillas débiles)
            white_weak = max(0.001, X[i, defense_indices[0]])  # Evitar división por cero
            black_weak = max(0.001, X[i, defense_indices[1]])
            
            defense_score_white = 1 / white_weak
            defense_score_black = 1 / black_weak
            total_defense = defense_score_white + defense_score_black
            
            if total_defense > 0:
                defense_score_white = defense_score_white / total_defense
                defense_score_black = defense_score_black / total_defense
            else:
                defense_score_white = defense_score_black = 0.5
            
            # Combinar puntuaciones ponderadas
            white_score = (
                weights['material'] * material_score_white +
                weights['activity'] * activity_score_white +
                weights['center'] * center_score_white +
                weights['attack'] * attack_score_white +
                weights['defense'] * defense_score_white
            )
            
            black_score = (
                weights['material'] * material_score_black +
                weights['activity'] * activity_score_black +
                weights['center'] * center_score_black +
                weights['attack'] * attack_score_black +
                weights['defense'] * defense_score_black
            )
            
            # Normalizar a 0-100
            white_performance[i] = white_score * 100
            black_performance[i] = black_score * 100
            
            # Ajustar según el resultado real
            result = y_result[i]
            if result == 0:  # Victoria blancas
                white_performance[i] = min(100, white_performance[i] * 1.2)
                black_performance[i] = max(0, black_performance[i] * 0.8)
            elif result == 1:  # Victoria negras
                white_performance[i] = max(0, white_performance[i] * 0.8)
                black_performance[i] = min(100, black_performance[i] * 1.2)
        
        # Añadir las etiquetas de rendimiento al DataFrame de características
        features_df['white_performance'] = white_performance
        features_df['black_performance'] = black_performance
        
        # Crear un DataFrame final combinando el DataFrame original con las características y etiquetas
        # Primero, reiniciar el índice del DataFrame original para asegurar la alineación
        df_reset = df.reset_index(drop=True)
        
        # Luego, concatenar con el DataFrame de características
        result_df = pd.concat([df_reset, features_df], axis=1)
        
        # Devolver el DataFrame completo
        return result_df
