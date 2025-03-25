import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chess
import chess.pgn
import io
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import warnings
from tqdm import tqdm
from evaluador import Evaluador

# Suprimir advertencias para una salida más limpia
warnings.filterwarnings('ignore')

class ChessResultPredictor:
    """
    Modelo de Machine Learning para predecir el resultado y rendimiento de partidas de ajedrez.
    Utiliza características extraídas de los movimientos y estado del tablero para realizar predicciones.
    """
    
    def __init__(self, models_dir='../models/chess_predictor'):
        """
        Inicializa el predictor de resultados de ajedrez.
        
        Args:
            models_dir: Directorio para guardar/cargar modelos
        """
        self.models_dir = models_dir
        self.result_model = None  # Modelo para predecir resultado (0 Gana Blanca, 1 Gana Negra, 0.5 Gana Empate)
        self.performance_model = None  # Modelo para predecir rendimiento (%)
        self.scaler = StandardScaler()
        self.evaluador = Evaluador()
        self.feature_names = None
        self.feature_importance = None
        
        # Crear directorio de modelos si no existe
        os.makedirs(models_dir, exist_ok=True)
    
    def prepare_data_from_dataframe(self, df, moves_col='Moves', result_col='Result', elo_cols=None, exclude_cols=None):
        """
        Prepara los datos para entrenamiento a partir de un DataFrame.
        
        Args:
            df: DataFrame con partidas
            moves_col: Nombre de la columna que contiene los movimientos
            result_col: Nombre de la columna que contiene el resultado
            elo_cols: Tupla con nombres de columnas de ELO (white_elo, black_elo)
            exclude_cols: Lista de columnas a excluir de las características
            
        Returns:
            X: Matriz de características
            y_result: Vector de resultados
            y_performance: Matriz de rendimiento [white_performance, black_performance]
        """
        print("Preparando datos para entrenamiento...")
        
        # Verificar si el DataFrame ya tiene características extraídas
        if 'white_performance' in df.columns and 'black_performance' in df.columns:
            print("Usando características ya extraídas del DataFrame")
            
            if exclude_cols is None:
                exclude_cols = []
            
            # Asegurarse de que las columnas de rendimiento y resultado no estén en las características
            exclude_cols.extend([result_col, 'white_performance', 'black_performance', moves_col])
            
            # Eliminar duplicados de la lista de exclusión
            exclude_cols = list(set(exclude_cols))
            
            # Seleccionar características (todas las columnas excepto las excluidas)
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Guardar nombres de características
            self.feature_names = feature_cols
            
            # Extraer características, resultados y rendimiento
            X = df[feature_cols].values
            y_result = df[result_col].values
            y_performance = df[['white_performance', 'black_performance']].values
            
            print(f"Datos preparados: {X.shape[0]} muestras, {X.shape[1]} características")
            
            return X, y_result, y_performance
        
        # Si no tiene características extraídas, usar el evaluador para extraerlas
        print("Extrayendo características usando el evaluador...")
        
        # Usar el evaluador para extraer características y calcular etiquetas de rendimiento
        df_con_caracteristicas = self.evaluador.calculate_performance_labels(
            df=df,
            moves_column=moves_col,
            n_jobs=-1,
            batch_size=1000
        )
        
        # Ahora que tenemos las características, llamar recursivamente a esta función
        return self.prepare_data_from_dataframe(
            df=df_con_caracteristicas,
            moves_col=moves_col,
            result_col=result_col,
            elo_cols=elo_cols,
            exclude_cols=exclude_cols
        )
    
    def extract_features_from_game(self, moves_str):
        """
        Extrae características de una partida a partir de una cadena de movimientos.
        
        Args:
            moves_str: String con los movimientos de la partida en formato algebraico.
        
        Returns:
            Un vector de características y una lista de nombres de características.
        """
        # Usar el evaluador para extraer características
        features_dict, feature_names = self.evaluador.extract_features_from_game(moves_str)
        
        # Convertir diccionario a vector
        features_vector = list(features_dict.values())
        
        return features_vector, feature_names
     
    def train_result_model(self, X, y, test_size=0.2, random_state=42, batch_size=10000):
        """
        Entrena un modelo para predecir el resultado de la partida.
        Optimizado para grandes conjuntos de datos.
        
        Args:
            X: Características
            y: Etiquetas (resultados)
            test_size: Proporción de datos para prueba
            random_state: Semilla aleatoria
            batch_size: Tamaño del lote para procesamiento
            
        Returns:
            X_test, y_test para evaluación
        """
        print("Entrenando modelo de predicción de resultado...")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Escalar características
        print("Escalando características...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Guardar scaler
        joblib.dump(self.scaler, f'{self.models_dir}/result_scaler.pkl')
        
        # Definir parámetros para búsqueda
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5]
        }
        
        # Crear modelo base
        base_model = XGBClassifier(
            objective='multi:softprob' if len(np.unique(y)) > 2 else 'binary:logistic',
            random_state=random_state,
            tree_method='hist',  # Más eficiente para grandes conjuntos de datos
            enable_categorical=False  # Evitar advertencias
        )
        
        # Realizar búsqueda de hiperparámetros con una muestra más pequeña para mayor velocidad
        print("Realizando búsqueda de hiperparámetros (esto puede tomar tiempo)...")
        
        # Usar una muestra para la búsqueda de hiperparámetros si el conjunto de datos es muy grande
        if X_train.shape[0] > 50000:
            sample_size = 50000
            print(f"Usando una muestra de {sample_size} ejemplos para búsqueda de hiperparámetros")
            indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
            X_sample = X_train_scaled[indices]
            y_sample = y_train[indices]
        else:
            X_sample = X_train_scaled
            y_sample = y_train
        
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=10,
            scoring='f1_weighted',
            cv=3,
            random_state=random_state,
            n_jobs=-1
        )
        
        search.fit(X_sample, y_sample)
        
        print(f"Mejores hiperparámetros: {search.best_params_}")
        
        # Entrenar modelo final con los mejores parámetros
        model = XGBClassifier(
            objective='multi:softprob' if len(np.unique(y)) > 2 else 'binary:logistic',
            **search.best_params_,
            random_state=random_state,
            tree_method='hist',
            enable_categorical=False,
            #early_stopping_rounds=10,
            eval_metric = "mlogloss"
        )
        
        # Para conjuntos de datos muy grandes, entrenar en lotes
        if X_train.shape[0] > batch_size:
            print(f"Entrenando en lotes de {batch_size} ejemplos...")
            # Inicializar modelo con el primer lote
            first_batch_size = min(batch_size, X_train.shape[0])
            model.fit(
                X_train_scaled[:first_batch_size], 
                y_train[:first_batch_size],
                eval_set=[(X_test_scaled, y_test)],  # Conjunto de evaluación
                verbose=True
            )
            
            # Continuar entrenando con el resto de lotes
            for start_idx in tqdm(range(first_batch_size, X_train.shape[0], batch_size), desc="Entrenando lotes"):
                end_idx = min(start_idx + batch_size, X_train.shape[0])
                model.fit(
                    X_train_scaled[start_idx:end_idx], 
                    y_train[start_idx:end_idx],
                    xgb_model=model,  # Continuar desde el modelo actual
                    eval_set=[(X_test_scaled, y_test)],
                    verbose=False
                )
        else:
            # Entrenar con todos los datos a la vez
            model.fit(
                X_train_scaled, 
                y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=True
            )
      
        # Guardar modelo
        self.result_model = model
        joblib.dump(model, f'{self.models_dir}/result_model.pkl')
        
        # Evaluar modelo
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Precisión del modelo: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Validación cruzada con una muestra si el conjunto de datos es muy grande
        if X.shape[0] > 50000:
            sample_size = 50000
            print(f"Realizando validación cruzada con una muestra de {sample_size} ejemplos...")
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
            cv_scores = cross_val_score(model, X_sample, y_sample, cv=3, scoring='f1_weighted')
        else:
            print("Realizando validación cruzada...")
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
            
        print(f"F1-score en validación cruzada: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Graficar matriz de confusión
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión - Predicción de Resultado')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.savefig(f'{self.models_dir}/result_confusion_matrix.png')
        plt.close()
        
        # Graficar importancia de características
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            feature_names = self.feature_names if self.feature_names else [f'Feature {i}' for i in range(X.shape[1])]
            
            # Ordenar características por importancia
            indices = np.argsort(model.feature_importances_)[::-1]
            top_indices = indices[:20]  # Mostrar solo las 20 más importantes
            
            # Guardar importancia de características
            self.feature_importance = {
                feature_names[i]: model.feature_importances_[i] for i in range(len(feature_names))
            }
            
            plt.barh(range(len(top_indices)), model.feature_importances_[top_indices])
            plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
            plt.title('Importancia de Características - Predicción de Resultado')
            plt.tight_layout()
            plt.savefig(f'{self.models_dir}/result_feature_importance.png')
            plt.close()
        
        return X_test_scaled, y_test
    
    def train_performance_model(self, X, y, test_size=0.2, random_state=42, batch_size=10000):
        """
        Entrena un modelo para predecir el rendimiento de los jugadores.
        Optimizado para grandes conjuntos de datos.
        
        Args:
            X: Características
            y: Etiquetas (rendimiento [white_performance, black_performance])
            test_size: Proporción de datos para prueba
            random_state: Semilla aleatoria
            batch_size: Tamaño del lote para procesamiento
            
        Returns:
            X_test, y_test para evaluación
        """
        print("Entrenando modelo de predicción de rendimiento...")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Escalar características si no se ha hecho ya
        if not hasattr(self, 'scaler') or self.scaler is None:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            joblib.dump(self.scaler, f'{self.models_dir}/performance_scaler.pkl')
        else:
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar modelo para rendimiento de blancas
        print("Entrenando modelo para rendimiento de blancas...")
        white_model = self._train_single_performance_model(
            X_train_scaled, y_train[:, 0], X_test_scaled, y_test[:, 0], 'white', batch_size
        )
        
        # Entrenar modelo para rendimiento de negras
        print("Entrenando modelo para rendimiento de negras...")
        black_model = self._train_single_performance_model(
            X_train_scaled, y_train[:, 1], X_test_scaled, y_test[:, 1], 'black', batch_size
        )
        
        # Crear modelo de red neuronal
        print("Entrenando red neuronal para rendimiento...")
        nn_model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(2)  # Salida: [white_performance, black_performance]
        ])
        
        # Compilar modelo
        optimizer = Adam(learning_rate=0.001)
        nn_model.compile(
            optimizer=optimizer,
            loss=MeanSquaredError(),
            metrics=['mae']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        model_checkpoint = ModelCheckpoint(
            filepath=f'{self.models_dir}/best_performance_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        # Para conjuntos de datos muy grandes, usar un subconjunto para validación
        if X_train.shape[0] > 50000:
            val_size = 10000
            print(f"Usando {val_size} ejemplos para validación")
            val_indices = np.random.choice(X_test_scaled.shape[0], val_size, replace=False)
            X_val = X_test_scaled[val_indices]
            y_val = y_test[val_indices]
        else:
            X_val = X_test_scaled
            y_val = y_test
        
        # Entrenar modelo
        if X_train.shape[0] > batch_size:
            print(f"Entrenando red neuronal en lotes de {batch_size} ejemplos...")
            # Inicializar historial de entrenamiento
            history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
            
            # Entrenar por épocas
            epochs = 50  # Reducido para mayor velocidad
            for epoch in range(epochs):
                print(f"Época {epoch+1}/{epochs}")
                
                # Mezclar datos en cada época
                indices = np.random.permutation(X_train_scaled.shape[0])
                X_train_shuffled = X_train_scaled[indices]
                y_train_shuffled = y_train[indices]
                
                # Entrenar en lotes
                batch_losses = []
                batch_maes = []
                for start_idx in tqdm(range(0, X_train_shuffled.shape[0], batch_size), desc="Lotes"):
                    end_idx = min(start_idx + batch_size, X_train_shuffled.shape[0])
                    X_batch = X_train_shuffled[start_idx:end_idx]
                    y_batch = y_train_shuffled[start_idx:end_idx]
                    
                    # Entrenar en este lote
                    batch_history = nn_model.fit(
                        X_batch, y_batch,
                        epochs=1,
                        verbose=0
                    )
                    
                    batch_losses.append(batch_history.history['loss'][0])
                    batch_maes.append(batch_history.history['mae'][0])
                
                # Evaluar en conjunto de validación
                val_metrics = nn_model.evaluate(X_val, y_val, verbose=0)
                
                # Actualizar historial
                epoch_loss = np.mean(batch_losses)
                epoch_mae = np.mean(batch_maes)
                history['loss'].append(epoch_loss)
                history['mae'].append(epoch_mae)
                history['val_loss'].append(val_metrics[0])
                history['val_mae'].append(val_metrics[1])
                
                print(f"loss: {epoch_loss:.4f} - mae: {epoch_mae:.4f} - val_loss: {val_metrics[0]:.4f} - val_mae: {val_metrics[1]:.4f}")
                
                # Implementar early stopping manualmente
                if len(history['val_loss']) > early_stopping.patience:
                    if val_metrics[0] > min(history['val_loss'][-(early_stopping.patience+1):-1]):
                        print(f"Early stopping en época {epoch+1}")
                        break
                
                # Guardar mejor modelo
                if val_metrics[0] == min(history['val_loss']):
                    print("Guardando mejor modelo...")
                    nn_model.save(f'{self.models_dir}/best_performance_model.keras')
        else:
            # Entrenar con todos los datos a la vez
            history = nn_model.fit(
                X_train_scaled,
                y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping, reduce_lr, model_checkpoint],
                verbose=1
            ).history
        
        # Cargar mejor modelo
        nn_model = load_model(f'{self.models_dir}/best_performance_model.keras')
        
        # Guardar modelo
        self.performance_model = nn_model
        nn_model.save(f'{self.models_dir}/performance_model.keras')
        
        # Evaluar modelo
        y_pred = nn_model.predict(X_test_scaled)
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = r2_score(y_test.flatten(), y_pred.flatten())
        
        print(f"Error cuadrático medio: {mse:.4f}")
        print(f"Error absoluto medio: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Graficar historial de entrenamiento
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Entrenamiento')
        plt.plot(history['val_loss'], label='Validación')
        plt.title('Pérdida durante el entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida (MSE)')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['mae'], label='Entrenamiento')
        plt.plot(history['val_mae'], label='Validación')
        plt.title('Error absoluto medio durante el entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.models_dir}/performance_training_history.png')
        plt.close()
        
        # Graficar predicciones vs valores reales (usar una muestra para mayor claridad)
        if X_test_scaled.shape[0] > 1000:
            sample_indices = np.random.choice(X_test_scaled.shape[0], 1000, replace=False)
            X_sample = X_test_scaled[sample_indices]
            y_sample = y_test[sample_indices]
            y_pred_sample = nn_model.predict(X_sample)
        else:
            y_sample = y_test
            y_pred_sample = y_pred
            
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_sample[:, 0], y_pred_sample[:, 0], alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.title('Rendimiento de Blancas: Real vs Predicho')
        plt.xlabel('Rendimiento Real')
        plt.ylabel('Rendimiento Predicho')
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_sample[:, 1], y_pred_sample[:, 1], alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.title('Rendimiento de Negras: Real vs Predicho')
        plt.xlabel('Rendimiento Real')
        plt.ylabel('Rendimiento Predicho')
        
        plt.tight_layout()
        plt.savefig(f'{self.models_dir}/performance_predictions.png')
        plt.close()
        
        # Comparar modelos
        self._compare_performance_models(X_test_scaled, y_test, white_model, black_model, nn_model)
        
        return X_test_scaled, y_test
    
    def _train_single_performance_model(self, X_train, y_train, X_test, y_test, color, batch_size=10000):
        """
        Entrena un modelo para predecir el rendimiento de un solo color.
        Optimizado para grandes conjuntos de datos.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de rendimiento (solo un color)
            X_test: Características de prueba
            y_test: Etiquetas de prueba (solo un color)
            color: 'white' o 'black'
            batch_size: Tamaño del lote para procesamiento
            
        Returns:
            Modelo entrenado
        """
        # Definir parámetros para búsqueda
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5]
        }
        
        # Crear modelo base
        base_model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            tree_method='hist',  # Más eficiente para grandes conjuntos de datos
            enable_categorical=False  # Evitar advertencias
        )
        
        # Realizar búsqueda de hiperparámetros con una muestra más pequeña para mayor velocidad
        print(f"Realizando búsqueda de hiperparámetros para {color}...")
        
        # Usar una muestra para la búsqueda de hiperparámetros si el conjunto de datos es muy grande
        if X_train.shape[0] > 50000:
            sample_size = 50000
            print(f"Usando una muestra de {sample_size} ejemplos para búsqueda de hiperparámetros")
            indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
            X_sample = X_train[indices]
            y_sample = y_train[indices]
        else:
            X_sample = X_train
            y_sample = y_train
        
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=10,
            scoring='neg_mean_squared_error',
            cv=3,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_sample, y_sample)
        
        print(f"Mejores hiperparámetros para {color}: {search.best_params_}")
        
        # Entrenar modelo final con los mejores parámetros
        model = XGBRegressor(
            objective='reg:squarederror',
            **search.best_params_,
            random_state=42,
            tree_method='hist',
            enable_categorical=False
        )
        
        # Para conjuntos de datos muy grandes, entrenar en lotes
        if X_train.shape[0] > batch_size:
            print(f"Entrenando en lotes de {batch_size} ejemplos...")
            # Inicializar modelo con el primer lote
            first_batch_size = min(batch_size, X_train.shape[0])
            model.fit(
                X_train[:first_batch_size], 
                y_train[:first_batch_size],
                eval_set=[(X_test, y_test)],
                #early_stopping_rounds=10,
                verbose=False
            )
            
            # Continuar entrenando con el resto de lotes
            for start_idx in tqdm(range(first_batch_size, X_train.shape[0], batch_size), desc=f"Entrenando lotes para {color}"):
                end_idx = min(start_idx + batch_size, X_train.shape[0])
                model.fit(
                    X_train[start_idx:end_idx], 
                    y_train[start_idx:end_idx],
                    xgb_model=model,  # Continuar desde el modelo actual
                    eval_set=[(X_test, y_test)],
                    #early_stopping_rounds=10,
                    verbose=False
                )
        else:
            # Entrenar con todos los datos a la vez
            model.fit(
                X_train, 
                y_train,
                eval_set=[(X_test, y_test)],
                #early_stopping_rounds=20,
                verbose=False
            )
        
        # Guardar modelo
        joblib.dump(model, f'{self.models_dir}/performance_{color}_model.pkl')
        
        # Evaluar modelo
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MSE para {color}: {mse:.4f}")
        print(f"R² para {color}: {r2:.4f}")
        
        # Graficar importancia de características
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            feature_names = self.feature_names if self.feature_names else [f'Feature {i}' for i in range(X_train.shape[1])]
            
            # Ordenar características por importancia
            indices = np.argsort(model.feature_importances_)[::-1]
            top_indices = indices[:20]  # Mostrar solo las 20 más importantes
            
            plt.barh(range(len(top_indices)), model.feature_importances_[top_indices])
            plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
            plt.title(f'Importancia de Características - Rendimiento de {color.capitalize()}')
            plt.tight_layout()
            plt.savefig(f'{self.models_dir}/performance_{color}_feature_importance.png')
            plt.close()
        
        return model
    
    def _compare_performance_models(self, X_test, y_test, white_model, black_model, nn_model):
        """
        Compara diferentes modelos de predicción de rendimiento.
        
        Args:
            X_test: Características de prueba
            y_test: Etiquetas de prueba [white_performance, black_performance]
            white_model: Modelo para rendimiento de blancas
            black_model: Modelo para rendimiento de negras
            nn_model: Modelo de red neuronal para ambos rendimientos
        """
        # Usar una muestra para la comparación si el conjunto de datos es muy grande
        if X_test.shape[0] > 1000:
            sample_size = 1000
            print(f"Usando una muestra de {sample_size} ejemplos para comparación de modelos")
            indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
            X_sample = X_test[indices]
            y_sample = y_test[indices]
        else:
            X_sample = X_test
            y_sample = y_test
        
        # Predecir con modelos individuales
        white_pred = white_model.predict(X_sample)
        black_pred = black_model.predict(X_sample)
        
        # Predecir con red neuronal
        nn_pred = nn_model.predict(X_sample)
        
        # Calcular métricas
        # Para blancas
        white_mse_xgb = mean_squared_error(y_sample[:, 0], white_pred)
        white_mse_nn = mean_squared_error(y_sample[:, 0], nn_pred[:, 0])
        
        # Para negras
        black_mse_xgb = mean_squared_error(y_sample[:, 1], black_pred)
        black_mse_nn = mean_squared_error(y_sample[:, 1], nn_pred[:, 1])
        
        # Graficar comparación
        plt.figure(figsize=(12, 10))
        
        # Blancas - XGBoost
        plt.subplot(2, 2, 1)
        plt.scatter(y_sample[:, 0], white_pred, alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.title(f'XGBoost - Rendimiento Blancas (MSE: {white_mse_xgb:.2f})')
        plt.xlabel('Real')
        plt.ylabel('Predicho')
        
        # Blancas - Red Neuronal
        plt.subplot(2, 2, 2)
        plt.scatter(y_sample[:, 0], nn_pred[:, 0], alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.title(f'Red Neuronal - Rendimiento Blancas (MSE: {white_mse_nn:.2f})')
        plt.xlabel('Real')
        plt.ylabel('Predicho')
        
        # Negras - XGBoost
        plt.subplot(2, 2, 3)
        plt.scatter(y_sample[:, 1], black_pred, alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.title(f'XGBoost - Rendimiento Negras (MSE: {black_mse_xgb:.2f})')
        plt.xlabel('Real')
        plt.ylabel('Predicho')
        
        # Negras - Red Neuronal
        plt.subplot(2, 2, 4)
        plt.scatter(y_sample[:, 1], nn_pred[:, 1], alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.title(f'Red Neuronal - Rendimiento Negras (MSE: {black_mse_nn:.2f})')
        plt.xlabel('Real')
        plt.ylabel('Predicho')
        
        plt.tight_layout()
        plt.savefig(f'{self.models_dir}/performance_models_comparison.png')
        plt.close()
        
        # Imprimir resultados
        print("\nComparación de modelos de rendimiento:")
        print(f"XGBoost - Rendimiento Blancas MSE: {white_mse_xgb:.4f}")
        print(f"Red Neuronal - Rendimiento Blancas MSE: {white_mse_nn:.4f}")
        print(f"XGBoost - Rendimiento Negras MSE: {black_mse_xgb:.4f}")
        print(f"Red Neuronal - Rendimiento Negras MSE: {black_mse_nn:.4f}")
    
    def train_models(self, df, moves_col='Moves', result_col='Result', elo_cols=None, exclude_cols=None, batch_size=10000):
        """
        Entrena ambos modelos (resultado y rendimiento) a partir de un DataFrame.
        
        Args:
            df: DataFrame con partidas
            moves_col: Nombre de la columna que contiene los movimientos
            result_col: Nombre de la columna que contiene el resultado
            elo_cols: Tupla con nombres de columnas de ELO (white_elo, black_elo)
            exclude_cols: Lista de columnas a excluir de las características
            batch_size: Tamaño del lote para procesamiento
            
        Returns:
            Diccionario con resultados de evaluación
        """
        # Preparar datos
        X, y_result, y_performance = self.prepare_data_from_dataframe(
            df, moves_col, result_col, elo_cols, exclude_cols
        )
        
        # Entrenar modelo de resultado
        X_test_result, y_test_result = self.train_result_model(X, y_result, batch_size=batch_size)
        
        # Entrenar modelo de rendimiento
        X_test_perf, y_test_perf = self.train_performance_model(X, y_performance, batch_size=batch_size)
        
        # Guardar nombres de características
        if self.feature_names:
            with open(f'{self.models_dir}/feature_names.txt', 'w') as f:
                for name in self.feature_names:
                    f.write(f"{name}\n")
        
        # Guardar importancia de características
        if self.feature_importance:
            with open(f'{self.models_dir}/feature_importance.txt', 'w') as f:
                for name, importance in sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{name}: {importance:.6f}\n")
        
        return {
            'X_test_result': X_test_result,
            'y_test_result': y_test_result,
            'X_test_performance': X_test_perf,
            'y_test_performance': y_test_perf
        }
    
    def load_models(self):
        """
        Carga modelos guardados.
        
        Returns:
            True si se cargaron correctamente, False en caso contrario
        """
        try:
            # Cargar modelo de resultado
            result_path = f'{self.models_dir}/result_model.pkl'
            if os.path.exists(result_path):
                self.result_model = joblib.load(result_path)
                print("Modelo de resultado cargado correctamente")
            else:
                print("No se encontró el modelo de resultado")
                return False
            
            # Cargar modelo de rendimiento
            performance_path = f'{self.models_dir}/performance_model.keras'
            if os.path.exists(performance_path):
                self.performance_model = load_model(performance_path)
                print("Modelo de rendimiento cargado correctamente")
            else:
                print("No se encontró el modelo de rendimiento")
                return False
            
            # Cargar scaler
            scaler_path = f'{self.models_dir}/result_scaler.pkl'
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("Scaler cargado correctamente")
            else:
                print("No se encontró el scaler")
                return False
            
            # Cargar nombres de características
            feature_names_path = f'{self.models_dir}/feature_names.txt'
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
                print(f"Nombres de características cargados: {len(self.feature_names)}")
            
            # Cargar importancia de características
            importance_path = f'{self.models_dir}/feature_importance.txt'
            if os.path.exists(importance_path):
                self.feature_importance = {}
                with open(importance_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(': ')
                        if len(parts) == 2:
                            self.feature_importance[parts[0]] = float(parts[1])
                print(f"Importancia de características cargada: {len(self.feature_importance)}")
            
            return True
        
        except Exception as e:
            print(f"Error al cargar modelos: {e}")
            return False
    
    def predict_game(self, moves_str):
        """
        Predice el resultado y rendimiento de una partida a partir de sus movimientos.
        
        Args:
            moves_str: String con los movimientos de la partida
            
        Returns:
            Diccionario con predicciones
        """
        if self.result_model is None or self.performance_model is None:
            if not self.load_models():
                print("No se pudieron cargar los modelos. Entrene los modelos primero.")
                return None
        
        # Extraer características
        features, _ = self.extract_features_from_game(moves_str)
        
        # Convertir a array y escalar
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        # Predecir resultado
        result_proba = self.result_model.predict_proba(X_scaled)[0]
        result_pred = self.result_model.predict(X_scaled)[0]
        
        # Predecir rendimiento
        performance_pred = self.performance_model.predict(X_scaled)[0]
        
        # Crear diccionario de resultados
        predictions = {
            'result': result_pred,
            'result_probabilities': {
                'white_win': result_proba[1] if len(result_proba) == 2 else result_proba[2],
                'draw': 0 if len(result_proba) == 2 else result_proba[1],
                'black_win': result_proba[0]
            },
            'performance': {
                'white': performance_pred[0],
                'black': performance_pred[1]
            }
        }
        
        return predictions
    
    def predict_dataframe(self, df, moves_col='Moves', batch_size=100):
        """
        Predice resultados y rendimiento para todas las partidas en un DataFrame.
        
        Args:
            df: DataFrame con partidas
            moves_col: Nombre de la columna que contiene los movimientos
            batch_size: Tamaño del lote para procesamiento
            
        Returns:
            DataFrame con predicciones añadidas
        """
        if self.result_model is None or self.performance_model is None:
            if not self.load_models():
                print("No se pudieron cargar los modelos. Entrene los modelos primero.")
                return df
        
        # Crear copia del DataFrame
        result_df = df.copy()
        
        # Añadir columnas para predicciones
        result_df['predicted_result'] = None
        result_df['white_win_probability'] = 0.0
        result_df['draw_probability'] = 0.0
        result_df['black_win_probability'] = 0.0
        result_df['white_performance'] = 0.0
        result_df['black_performance'] = 0.0
        
        # Procesar en lotes
        total_batches = (len(result_df) + batch_size - 1) // batch_size
        
        for i in range(0, len(result_df), batch_size):
            end_idx = min(i + batch_size, len(result_df))
            batch_num = i // batch_size + 1
            print(f"Procesando lote {batch_num}/{total_batches} (partidas {i+1} a {end_idx} de {len(result_df)})...")
            
            # Extraer características para este lote
            features_list = []
            for idx, row in result_df.iloc[i:end_idx].iterrows():
                try:
                    features, _ = self.extract_features_from_game(row[moves_col])
                    features_list.append(features)
                except Exception as e:
                    print(f"Error al procesar partida {idx}: {e}")
                    # Usar características vacías (ceros) en caso de error
                    if self.feature_names:
                        features_list.append([0] * len(self.feature_names))
                    else:
                        # Estimar número de características basado en la primera partida
                        if len(features_list) > 0:
                            features_list.append([0] * len(features_list[0]))
                        else:
                            # Si es la primera partida y hay error, usar un valor predeterminado
                            features_list.append([0] * 36)  # Número aproximado de características
            
            # Verificar que hay características para procesar
            if not features_list:
                print(f"No se pudieron extraer características para el lote {batch_num}. Saltando...")
                continue
            
            # Convertir a array y escalar
            X = np.array(features_list)
            X_scaled = self.scaler.transform(X)
            
            # Predecir resultados
            result_proba = self.result_model.predict_proba(X_scaled)
            result_pred = self.result_model.predict(X_scaled)
            
            # Predecir rendimiento
            performance_pred = self.performance_model.predict(X_scaled)
            
            # Guardar predicciones
            for j, idx in enumerate(range(i, end_idx)):
                if j >= len(result_pred):  # Verificar índices
                    break
                    
                result_df.at[idx, 'predicted_result'] = result_pred[j]
                
                # Probabilidades
                if len(result_proba[j]) == 2:  # Binario (0, 1)
                    result_df.at[idx, 'black_win_probability'] = result_proba[j][0]
                    result_df.at[idx, 'draw_probability'] = 0.0
                    result_df.at[idx, 'white_win_probability'] = result_proba[j][1]
                else:  # Multiclase (0, 0.5, 1)
                    result_df.at[idx, 'black_win_probability'] = result_proba[j][0]
                    result_df.at[idx, 'draw_probability'] = result_proba[j][1]
                    result_df.at[idx, 'white_win_probability'] = result_proba[j][2]
                
                # Rendimiento
                result_df.at[idx, 'white_performance'] = performance_pred[j][0]
                result_df.at[idx, 'black_performance'] = performance_pred[j][1]
        
        return result_df
    
    def calculate_elo_change(self, white_elo, black_elo, result):
        """
        Calcula el cambio de ELO para ambos jugadores basado en el resultado de la partida.
        
        Args:
            white_elo: ELO del jugador de blancas
            black_elo: ELO del jugador de negras
            result: Resultado de la partida (1=victoria blancas, 0.5=tablas, 0=victoria negras)
            
        Returns:
            Tupla con (cambio_elo_blancas, cambio_elo_negras)
        """
        # Constante K (factor de desarrollo)
        # Valores típicos: 40 para jugadores nuevos, 20 para jugadores establecidos, 10 para maestros
        k_factor = 20
        
        # Calcular expectativa de victoria
        elo_diff = white_elo - black_elo
        expected_white = 1 / (1 + 10 ** (-elo_diff / 400))
        expected_black = 1 - expected_white
        
        # Calcular cambio de ELO
        white_change = k_factor * (result - expected_white)
        black_change = k_factor * ((1 - result) - expected_black)
        
        return white_change, black_change
    
    def analyze_game(self, moves_str, white_elo=None, black_elo=None):
        """
        Realiza un análisis detallado de una partida.
        
        Args:
            moves_str: String con los movimientos de la partida
            white_elo: ELO del jugador de blancas (opcional)
            black_elo: ELO del jugador de negras (opcional)
            
        Returns:
            Diccionario con análisis detallado
        """
        # Predecir resultado y rendimiento
        predictions = self.predict_game(moves_str)
        if not predictions:
            return None
        
        # Extraer características
        features, feature_names = self.extract_features_from_game(moves_str)
        
        # Crear diccionario de características
        features_dict = {name: value for name, value in zip(feature_names, features)}
        
        # Calcular ventaja material
        white_material = (
            features_dict['white_pawns_avg'] + 
            features_dict['white_knights_avg'] * 3 + 
            features_dict['white_bishops_avg'] * 3 + 
            features_dict['white_rooks_avg'] * 5 + 
            features_dict['white_queens_avg'] * 9
        )
        
        black_material = (
            features_dict['black_pawns_avg'] + 
            features_dict['black_knights_avg'] * 3 + 
            features_dict['black_bishops_avg'] * 3 + 
            features_dict['black_rooks_avg'] * 5 + 
            features_dict['black_queens_avg'] * 9
        )
        
        material_advantage = white_material - black_material
        
        # Determinar fase de la partida
        total_pieces = sum([
            features_dict['white_pawns_avg'],
            features_dict['black_pawns_avg'],
            features_dict['white_knights_avg'],
            features_dict['black_knights_avg'],
            features_dict['white_bishops_avg'],
            features_dict['black_bishops_avg'],
            features_dict['white_rooks_avg'],
            features_dict['black_rooks_avg'],
            features_dict['white_queens_avg'],
            features_dict['black_queens_avg']
        ])
        
        if total_pieces >= 28:
            game_phase = "Apertura"
        elif total_pieces >= 20:
            game_phase = "Medio juego"
        else:
            game_phase = "Final"
        
        # Determinar ventaja posicional
        white_positional = (
            features_dict['white_center_control_avg'] * 0.3 +
            features_dict['white_mobility_avg'] * 0.3 +
            features_dict['white_king_safety_avg'] * 0.2 +
            features_dict['white_piece_activity_avg'] * 0.2
        )
        
        black_positional = (
            features_dict['black_center_control_avg'] * 0.3 +
            features_dict['black_mobility_avg'] * 0.3 +
            features_dict['black_king_safety_avg'] * 0.2 +
            features_dict['black_piece_activity_avg'] * 0.2
        )
        
        positional_advantage = white_positional - black_positional
        
        # Determinar jugador con ventaja
        if material_advantage > 1:
            material_advantage_player = "Blancas"
        elif material_advantage < -1:
            material_advantage_player = "Negras"
        else:
            material_advantage_player = "Igualdad"
        
        if positional_advantage > 2:
            positional_advantage_player = "Blancas"
        elif positional_advantage < -2:
            positional_advantage_player = "Negras"
        else:
            positional_advantage_player = "Igualdad"
        
        # Calcular cambio de ELO si se proporcionaron valores de ELO
        elo_changes = None
        if white_elo is not None and black_elo is not None:
            # Calcular cambio de ELO para el resultado predicho
            predicted_white_change, predicted_black_change = self.calculate_elo_change(
                white_elo, black_elo, predictions['result']
            )
            
            # Calcular cambio de ELO para cada posible resultado
            white_win_changes = self.calculate_elo_change(white_elo, black_elo, 1)
            draw_changes = self.calculate_elo_change(white_elo, black_elo, 0.5)
            black_win_changes = self.calculate_elo_change(white_elo, black_elo, 0)
            
            elo_changes = {
                'predicted': {
                    'white': predicted_white_change,
                    'black': predicted_black_change
                },
                'if_white_wins': {
                    'white': white_win_changes[0],
                    'black': white_win_changes[1]
                },
                'if_draw': {
                    'white': draw_changes[0],
                    'black': draw_changes[1]
                },
                'if_black_wins': {
                    'white': black_win_changes[0],
                    'black': black_win_changes[1]
                }
            }
        
        # Crear análisis
        analysis = {
            'predictions': predictions,
            'game_phase': game_phase,
            'material': {
                'white': white_material,
                'black': black_material,
                'advantage': material_advantage,
                'advantage_player': material_advantage_player
            },
            'positional': {
                'white': white_positional,
                'black': black_positional,
                'advantage': positional_advantage,
                'advantage_player': positional_advantage_player
            },
            'development': {
                'white_time': features_dict['white_development_time'],
                'black_time': features_dict['black_development_time'],
                'white_castled': bool(features_dict['white_castled']),
                'black_castled': bool(features_dict['black_castled'])
            },
            'activity': {
                'white_center_control': features_dict['white_center_control_avg'],
                'black_center_control': features_dict['black_center_control_avg'],
                'white_mobility': features_dict['white_mobility_avg'],
                'black_mobility': features_dict['black_mobility_avg'],
                'white_attacks': features_dict['white_attacks_avg'],
                'black_attacks': features_dict['black_attacks_avg']
            },
            'pawn_structure': {
                'white_doubled': features_dict['white_doubled_pawns_avg'],
                'black_doubled': features_dict['black_doubled_pawns_avg'],
                'white_isolated': features_dict['white_isolated_pawns_avg'],
                'black_isolated': features_dict['black_isolated_pawns_avg']
            },
            'key_statistics': {
                'total_moves': features_dict['total_moves'],
                'white_checks': features_dict['white_checks'],
                'black_checks': features_dict['black_checks'],
                'white_captures': features_dict['white_captures'],
                'black_captures': features_dict['black_captures']
            }
        }
        
        # Añadir información de ELO si está disponible
        if elo_changes:
            analysis['elo'] = {
                'white': white_elo,
                'black': black_elo,
                'changes': elo_changes
            }
        
        return analysis
    
    def generate_game_report(self, moves_str, output_file=None, white_elo=None, black_elo=None):
        """
        Genera un informe detallado de una partida en formato texto.
        
        Args:
            moves_str: String con los movimientos de la partida
            output_file: Ruta del archivo para guardar el informe (opcional)
            white_elo: ELO del jugador de blancas (opcional)
            black_elo: ELO del jugador de negras (opcional)
            
        Returns:
            String con el informe
        """
        # Realizar análisis
        analysis = self.analyze_game(moves_str, white_elo, black_elo)
        if not analysis:
            return "No se pudo analizar la partida."
        
        # Crear informe
        report = []
        report.append("=" * 60)
        report.append("INFORME DE ANÁLISIS DE PARTIDA DE AJEDREZ")
        report.append("=" * 60)
        report.append("")
        
        # Información de ELO si está disponible
        if 'elo' in analysis:
            report.append("INFORMACIÓN DE ELO:")
            report.append(f"  ELO Blancas: {analysis['elo']['white']}")
            report.append(f"  ELO Negras: {analysis['elo']['black']}")
            report.append("")
        
        # Predicciones
        pred = analysis['predictions']
        report.append("PREDICCIÓN DE RESULTADO:")
        result_text = {1: "Victoria de blancas", 0.5: "Tablas", 0: "Victoria de negras"}
        report.append(f"  Resultado más probable: {result_text.get(pred['result'], pred['result'])}")
        report.append(f"  Probabilidad de victoria blancas: {pred['result_probabilities']['white_win']:.2%}")
        report.append(f"  Probabilidad de tablas: {pred['result_probabilities']['draw']:.2%}")
        report.append(f"  Probabilidad de victoria negras: {pred['result_probabilities']['black_win']:.2%}")
        report.append("")
        
        # Cambio de ELO si está disponible
        if 'elo' in analysis:
            report.append("CAMBIO DE ELO ESPERADO:")
            elo_changes = analysis['elo']['changes']
            
            report.append(f"  Para el resultado predicho ({result_text.get(pred['result'], pred['result'])}):")
            report.append(f"    Blancas: {elo_changes['predicted']['white']:+.1f} puntos")
            report.append(f"    Negras: {elo_changes['predicted']['black']:+.1f} puntos")
            report.append("")
            
            report.append("  Cambios de ELO para cada posible resultado:")
            report.append(f"    Si ganan blancas: Blancas {elo_changes['if_white_wins']['white']:+.1f}, Negras {elo_changes['if_white_wins']['black']:+.1f}")
            report.append(f"    Si hay tablas: Blancas {elo_changes['if_draw']['white']:+.1f}, Negras {elo_changes['if_draw']['black']:+.1f}")
            report.append(f"    Si ganan negras: Blancas {elo_changes['if_black_wins']['white']:+.1f}, Negras {elo_changes['if_black_wins']['black']:+.1f}")
            report.append("")
        
        # Rendimiento
        report.append("RENDIMIENTO DE LOS JUGADORES:")
        report.append(f"  Rendimiento de blancas: {pred['performance']['white']:.1f}/100")
        report.append(f"  Rendimiento de negras: {pred['performance']['black']:.1f}/100")
        report.append("")
        
        # Fase de la partida
        report.append(f"FASE DE LA PARTIDA: {analysis['game_phase']}")
        report.append(f"  Total de movimientos: {analysis['key_statistics']['total_moves']}")
        report.append("")
        
        # Ventaja material
        report.append("BALANCE MATERIAL:")
        report.append(f"  Material blancas: {analysis['material']['white']:.1f}")
        report.append(f"  Material negras: {analysis['material']['black']:.1f}")
        report.append(f"  Ventaja material: {abs(analysis['material']['advantage']):.1f} a favor de {analysis['material']['advantage_player']}")
        report.append("")
        
        # Ventaja posicional
        report.append("BALANCE POSICIONAL:")
        report.append(f"  Puntuación posicional blancas: {analysis['positional']['white']:.1f}")
        report.append(f"  Puntuación posicional negras: {analysis['positional']['black']:.1f}")
        report.append(f"  Ventaja posicional: {abs(analysis['positional']['advantage']):.1f} a favor de {analysis['positional']['advantage_player']}")
        report.append("")
        
        # Desarrollo
        report.append("DESARROLLO:")
        report.append(f"  Tiempo de desarrollo blancas: {analysis['development']['white_time']} movimientos")
        report.append(f"  Tiempo de desarrollo negras: {analysis['development']['black_time']} movimientos")
        report.append(f"  Enroque blancas: {'Sí' if analysis['development']['white_castled'] else 'No'}")
        report.append(f"  Enroque negras: {'Sí' if analysis['development']['black_castled'] else 'No'}")
        report.append("")
        
        # Actividad
        report.append("ACTIVIDAD:")
        report.append(f"  Control del centro blancas: {analysis['activity']['white_center_control']:.1f}")
        report.append(f"  Control del centro negras: {analysis['activity']['black_center_control']:.1f}")
        report.append(f"  Movilidad blancas: {analysis['activity']['white_mobility']:.1f}")
        report.append(f"  Movilidad negras: {analysis['activity']['black_mobility']:.1f}")
        report.append(f"  Ataques blancas: {analysis['activity']['white_attacks']:.1f}")
        report.append(f"  Ataques negras: {analysis['activity']['black_attacks']:.1f}")
        report.append("")
        
        # Estructura de peones
        report.append("ESTRUCTURA DE PEONES:")
        report.append(f"  Peones doblados blancas: {analysis['pawn_structure']['white_doubled']:.1f}")
        report.append(f"  Peones doblados negras: {analysis['pawn_structure']['black_doubled']:.1f}")
        report.append(f"  Peones aislados blancas: {analysis['pawn_structure']['white_isolated']:.1f}")
        report.append(f"  Peones aislados negras: {analysis['pawn_structure']['black_isolated']:.1f}")
        report.append("")
        
        # Estadísticas clave
        report.append("ESTADÍSTICAS CLAVE:")
        report.append(f"  Capturas blancas: {analysis['key_statistics']['white_captures']}")
        report.append(f"  Capturas negras: {analysis['key_statistics']['black_captures']}")
        report.append(f"  Jaques blancas: {analysis['key_statistics']['white_checks']}")
        report.append(f"  Jaques negras: {analysis['key_statistics']['black_checks']}")
        report.append("")
        
        # Conclusión
        report.append("CONCLUSIÓN:")
        if pred['result'] == 1:
            if pred['result_probabilities']['white_win'] > 0.7:
                conclusion = "Las blancas tienen una ventaja decisiva."
            elif pred['result_probabilities']['white_win'] > 0.6:
                conclusion = "Las blancas tienen una ventaja clara."
            else:
                conclusion = "Las blancas tienen una ligera ventaja."
        elif pred['result'] == 0:
            if pred['result_probabilities']['black_win'] > 0.7:
                conclusion = "Las negras tienen una ventaja decisiva."
            elif pred['result_probabilities']['black_win'] > 0.6:
                conclusion = "Las negras tienen una ventaja clara."
            else:
                conclusion = "Las negras tienen una ligera ventaja."
        else:
            if pred['result_probabilities']['draw'] > 0.7:
                conclusion = "La posición está muy equilibrada y probablemente termine en tablas."
            else:
                conclusion = "La posición está ligeramente equilibrada, pero podría inclinarse hacia cualquier lado."
        
        report.append(f"  {conclusion}")
        report.append("")
        report.append("=" * 60)
        
        # Unir informe
        report_text = "\n".join(report)
        
        # Guardar en archivo si se especificó
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Informe guardado en {output_file}")
        
        return report_text

# Ejemplo de uso
if __name__ == "__main__":
    # Crear datos de ejemplo
    data = {
        'Event': ['Game 1', 'Game 2', 'Game 3'],
        'Result': [1, 0.5, 0],  # 1=blancas ganan, 0.5=tablas, 0=negras ganan
        'WhiteElo': [1800, 2000, 1600],
        'BlackElo': [1700, 2100, 1900],
        'Moves': [
            'e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 O-O d6 c3 O-O',
            'd4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6',
            'e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Crear predictor
    predictor = ChessResultPredictor(models_dir='../models/chess_predictor')
    
    # Entrenar modelos
    eval_results = predictor.train_models(
        df, 
        moves_col='Moves', 
        result_col='Result',
        elo_cols=('WhiteElo', 'BlackElo')
    )
    
    # Predecir para una nueva partida
    new_game = 'e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O'
    predictions = predictor.predict_game(new_game)
    
    print("\nPredicciones para nueva partida:")
    print(f"Resultado predicho: {predictions['result']}")
    print(f"Probabilidades: Blancas {predictions['result_probabilities']['white_win']:.2f}, " +
          f"Tablas {predictions['result_probabilities']['draw']:.2f}, " +
          f"Negras {predictions['result_probabilities']['black_win']:.2f}")
    print(f"Rendimiento: Blancas {predictions['performance']['white']:.2f}, " +
          f"Negras {predictions['performance']['black']:.2f}")
    
    # Generar informe detallado con información de ELO
    report = predictor.generate_game_report(new_game, 'game_report.txt', white_elo=1850, black_elo=1750)
    print("\nInforme generado y guardado en 'game_report.txt'")
    
    # Predecir para todo el DataFrame
    df_with_predictions = predictor.predict_dataframe(df)
    
    print("\nDataFrame con predicciones:")
    print(df_with_predictions[['Event', 'Result', 'predicted_result', 
                              'white_performance', 'black_performance']])
