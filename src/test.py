import pandas as pd
import numpy as np
from Ajedrez_Prediccion import ChessResultPredictor

# Ejemplo de uso con un pequeño conjunto de datos
def test_chess_predictor():
    # Crear datos de ejemplo
    data = {
        'Event': ['Game 1', 'Game 2', 'Game 3'],
        'Result': [0, 1, 2],  # 0=blancas ganan, 2=tablas, 1=negras ganan
        'WhiteElo': [1800, 2000, 1600],
        'BlackElo': [1700, 2100, 1900],
        'Moves': [
            'e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 O-O d6 c3 O-O',
            'd4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6',
            'e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5'
        ]
    }
    
    df = pd.DataFrame(data)
    print("DataFrame de prueba creado con éxito.")
    
    # Crear predictor
    predictor = ChessResultPredictor(models_dir='../models/test_chess_predictor')
    print("Instancia de ChessResultPredictor creada correctamente.")
    
    # Extraer características de una partida
    print("\nProbando extracción de características...")
    features, feature_names = predictor.extract_features_from_game(df['Moves'][0])
    print(f"Características extraídas: {len(features)}")
    print(f"Nombres de características: {len(feature_names)}")
    
    # Preparar datos
    print("\nProbando preparación de datos...")
    X, y_result, y_performance = predictor.prepare_data_from_dataframe(
        df, 
        moves_col='Moves', 
        result_col='Result',
        elo_cols=('WhiteElo', 'BlackElo')
    )
    print(f"Matriz X: {X.shape}")
    print(f"Vector y_result: {y_result.shape}")
    print(f"Matriz y_performance: {y_performance.shape}")
    
    print("\nPrueba completada con éxito.")
    return predictor

if __name__ == "__main__":
    test_chess_predictor()
