import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import chess
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import NotFittedError
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import warnings

# Suprimir advertencias para una salida más limpia
warnings.filterwarnings('ignore')

class ChessModelEnsemble:
    """
    Clase unificada para entrenamiento, evaluación y predicción de modelos de ajedrez.
    Integra Random Forest, Voting Classifier y Red Neuronal en un solo ensemble.
    """
    
    def __init__(self, random_state=42, models_dir='Models'):
        """
        Inicializa el ensemble de modelos.
        
        Args:
            random_state: Semilla para reproducibilidad
            models_dir: Directorio para guardar/cargar modelos
        """
        self.random_state = random_state
        self.models_dir = models_dir
        self.rf_model = None
        self.voting_model = None
        self.nn_model = None
        self.gb_model = None  # Añadido modelo Gradient Boosting
        self.ensemble_weights = [0.35, 0.25, 0.25, 0.15]  # Pesos para RF, Voting, NN, GB
        self.feature_names = None
        self.scaler = StandardScaler()
        self.model_metrics = {}  # Para almacenar métricas de rendimiento
        
        # Crear directorio de modelos si no existe
        os.makedirs(models_dir, exist_ok=True)
    
    def preprocess_data(self, df, target_col='Result', test_size=0.2):
        """
        Preprocesa los datos para entrenamiento y prueba.
        
        Args:
            df: DataFrame con los datos
            target_col: Nombre de la columna objetivo
            test_size: Proporción de datos para prueba
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("Preprocesando datos...")
        
        # Guardar nombres de características para uso posterior
        X = df.drop(columns=[target_col])
        self.feature_names = X.columns.tolist()
        y = df[target_col]
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Datos divididos: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} muestras de prueba")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, n_iter=50, cv=3, n_jobs=-1):
        """
        Entrena un modelo de Random Forest optimizado.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            n_iter: Número de iteraciones para búsqueda aleatoria
            cv: Número de folds para validación cruzada
            n_jobs: Número de trabajos paralelos (-1 para todos los núcleos)
            
        Returns:
            Modelo entrenado
        """
        print("Entrenando modelo Random Forest...")
        start_time = time.time()
        
        # Crear pipeline con escalado
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=self.random_state))
        ])
        
        # Parámetros optimizados (reducidos para mayor eficiencia)
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [20, 40, 60, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__bootstrap': [True, False],
            'classifier__class_weight': ['balanced', 'balanced_subsample', None]
        }
        
        # Búsqueda aleatoria con menos iteraciones para mayor eficiencia
        rf_random = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            verbose=1,
            random_state=self.random_state,
            n_jobs=n_jobs,
            scoring='f1'  # Optimizar para F1-score
        )
        
        # Entrenamiento
        rf_random.fit(X_train, y_train)
        
        # Guardar modelo
        self.rf_model = rf_random
        joblib.dump(rf_random, f'{self.models_dir}/RandomForestModel.pkl')
        
        training_time = time.time() - start_time
        print(f"Entrenamiento completado en {training_time:.2f} segundos")
        print(f"Mejores parámetros: {rf_random.best_params_}")
        
        return rf_random
    
    def train_gradient_boosting(self, X_train, y_train, n_iter=30, cv=3, n_jobs=-1):
        """
        Entrena un modelo de Gradient Boosting optimizado.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            n_iter: Número de iteraciones para búsqueda aleatoria
            cv: Número de folds para validación cruzada
            n_jobs: Número de trabajos paralelos (-1 para todos los núcleos)
            
        Returns:
            Modelo entrenado
        """
        print("Entrenando modelo Gradient Boosting...")
        start_time = time.time()
        
        # Crear pipeline con escalado
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(random_state=self.random_state))
        ])
        
        # Parámetros para búsqueda
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 5, 7],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__subsample': [0.8, 0.9, 1.0]
        }
        
        # Búsqueda aleatoria
        gb_random = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            verbose=1,
            random_state=self.random_state,
            n_jobs=n_jobs,
            scoring='f1'
        )
        
        # Entrenamiento
        gb_random.fit(X_train, y_train)
        
        # Guardar modelo
        self.gb_model = gb_random
        joblib.dump(gb_random, f'{self.models_dir}/GradientBoostingModel.pkl')
        
        training_time = time.time() - start_time
        print(f"Entrenamiento completado en {training_time:.2f} segundos")
        print(f"Mejores parámetros: {gb_random.best_params_}")
        
        return gb_random
    
    def train_voting_classifier(self, X_train, y_train):
        """
        Entrena un modelo de votación con LogisticRegression y RandomForest.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            
        Returns:
            Modelo entrenado
        """
        print("Entrenando modelo de Votación...")
        start_time = time.time()
        
        # Crear pipeline para LogisticRegression
        log_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000,
                class_weight='balanced',
                C=1.0
            ))
        ])
        
        # Crear pipeline para RandomForest (usando parámetros optimizados si están disponibles)
        if self.rf_model is not None and hasattr(self.rf_model, 'best_params_'):
            best_params = {k.replace('classifier__', ''): v 
                          for k, v in self.rf_model.best_params_.items()}
            rf_clf = RandomForestClassifier(random_state=self.random_state, **best_params)
        else:
            rf_clf = RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=200,
                class_weight='balanced'
            )
        
        rf_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', rf_clf)
        ])
        
        # Crear pipeline para GradientBoosting (si está disponible)
        if self.gb_model is not None and hasattr(self.gb_model, 'best_params_'):
            best_params = {k.replace('classifier__', ''): v 
                          for k, v in self.gb_model.best_params_.items()}
            gb_clf = GradientBoostingClassifier(random_state=self.random_state, **best_params)
        else:
            gb_clf = GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=200
            )
        
        gb_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', gb_clf)
        ])
        
        # Crear modelo de votación
        voting_clf = VotingClassifier(
            estimators=[('lr', log_pipe), ('rf', rf_pipe), ('gb', gb_pipe)],
            voting='soft'
        )
        
        # Entrenamiento
        voting_clf.fit(X_train, y_train)
        
        # Guardar modelo
        self.voting_model = voting_clf
        joblib.dump(voting_clf, f'{self.models_dir}/VotingModel.pkl')
        
        training_time = time.time() - start_time
        print(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        return voting_clf
    
    def train_neural_network(self, X_train, y_train, X_test, y_test, epochs=50):
        """
        Entrena un modelo de red neuronal optimizado.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_test: Características de prueba
            y_test: Etiquetas de prueba
            epochs: Número máximo de épocas
            
        Returns:
            Modelo entrenado e historial de entrenamiento
        """
        print("Entrenando modelo de Red Neuronal...")
        start_time = time.time()
        
        # Escalar datos
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Guardar el scaler para uso futuro
        joblib.dump(self.scaler, f'{self.models_dir}/scaler.pkl')
        
        # Callbacks para optimización
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            filepath=f'{self.models_dir}/best_nn_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Arquitectura mejorada
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        
        # Compilar con optimizador Adam y tasa de aprendizaje personalizada
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Entrenamiento con callbacks
        history = model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=1,
            callbacks=[early_stopping, reduce_lr, model_checkpoint]
        )
        
        # Cargar el mejor modelo guardado
        model = load_model(f'{self.models_dir}/best_nn_model.h5')
        
        # Guardar modelo final
        self.nn_model = model
        model.save(f'{self.models_dir}/NeuralNetworkModel.h5')
        
        training_time = time.time() - start_time
        print(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        # Graficar historial de entrenamiento
        self._plot_training_history(history)
        
        return model, history
    
    def _plot_training_history(self, history):
        """
        Genera gráficos del historial de entrenamiento de la red neuronal.
        
        Args:
            history: Historial de entrenamiento de Keras
        """
        plt.figure(figsize=(12, 5))
        
        # Gráfico de pérdida
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Entrenamiento')
        plt.plot(history.history['val_loss'], label='Validación')
        plt.title('Pérdida durante el entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        
        # Gráfico de precisión
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Entrenamiento')
        plt.plot(history.history['val_accuracy'], label='Validación')
        plt.title('Precisión durante el entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.models_dir}/nn_training_history.png')
        plt.close()
    
    def train_all_models(self, df, target_col='Result', test_size=0.2):
        """
        Entrena todos los modelos en secuencia.
        
        Args:
            df: DataFrame con los datos
            target_col: Nombre de la columna objetivo
            test_size: Proporción de datos para prueba
            
        Returns:
            X_test, y_test para evaluación
        """
        # Preprocesar datos
        X_train, X_test, y_train, y_test = self.preprocess_data(
            df, target_col, test_size
        )
        
        # Entrenar modelos
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        self.train_voting_classifier(X_train, y_train)
        self.train_neural_network(X_train, y_train, X_test, y_test)
        
        return X_test, y_test
    
    def load_models(self):
        """
        Carga todos los modelos guardados.
        
        Returns:
            True si todos los modelos se cargaron correctamente, False en caso contrario
        """
        try:
            print("Cargando modelos guardados...")
            models_loaded = 0
            
            # Cargar Random Forest
            rf_path = f'{self.models_dir}/RandomForestModel.pkl'
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                models_loaded += 1
                print("- Modelo Random Forest cargado")
            
            # Cargar Gradient Boosting
            gb_path = f'{self.models_dir}/GradientBoostingModel.pkl'
            if os.path.exists(gb_path):
                self.gb_model = joblib.load(gb_path)
                models_loaded += 1
                print("- Modelo Gradient Boosting cargado")
            
            # Cargar Voting Classifier
            voting_path = f'{self.models_dir}/VotingModel.pkl'
            if os.path.exists(voting_path):
                self.voting_model = joblib.load(voting_path)
                models_loaded += 1
                print("- Modelo Voting Classifier cargado")
            
            # Cargar Red Neuronal
            nn_path = f'{self.models_dir}/NeuralNetworkModel.h5'
            if os.path.exists(nn_path):
                self.nn_model = load_model(nn_path)
                models_loaded += 1
                print("- Modelo Red Neuronal cargado")
            
            # Cargar scaler
            scaler_path = f'{self.models_dir}/scaler.pkl'
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("- Scaler cargado")
            
            print(f"Se cargaron {models_loaded} de 4 modelos")
            return models_loaded > 0
        
        except Exception as e:
            print(f"Error al cargar modelos: {e}")
            return False
    
    def evaluate_model(self, model, X_test, y_test, model_name, plot=True):
        """
        Evalúa un modelo y muestra métricas.
        
        Args:
            model: Modelo a evaluar
            X_test: Características de prueba
            y_test: Etiquetas de prueba
            model_name: Nombre del modelo para mostrar
            plot: Si se debe mostrar la matriz de confusión
            
        Returns:
            Predicciones del modelo
        """
        print(f"\n--- Evaluación del modelo: {model_name} ---")
        
        # Escalar datos si es necesario (para red neuronal)
        if model_name == 'Neural Network':
            X_test_scaled = self.scaler.transform(X_test)
            y_pred_proba = model.predict(X_test_scaled)
            y_pred = np.where(y_pred_proba > 0.5, 1, 0)
        else:
            y_pred = model.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Guardar métricas
        self.model_metrics[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Mostrar reporte de clasificación
        print(classification_report(y_test, y_pred))
        
        # Mostrar matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        if plot:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis')
            plt.title(f'Matriz de Confusión - {model_name}', fontsize=16)
            plt.xlabel('Predicciones', fontsize=12)
            plt.ylabel('Valores Verdaderos', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{self.models_dir}/{model_name.replace(" ", "_")}_confusion.png')
            plt.close()
        
        return y_pred
    
    def evaluate_all_models(self, X_test, y_test, plot=True):
        """
        Evalúa todos los modelos entrenados.
        
        Args:
            X_test: Características de prueba
            y_test: Etiquetas de prueba
            plot: Si se deben mostrar gráficos
            
        Returns:
            Diccionario con predicciones de cada modelo
        """
        predictions = {}
        
        # Evaluar Random Forest
        if self.rf_model:
            rf_preds = self.evaluate_model(
                self.rf_model, X_test, y_test, 'Random Forest', plot
            )
            predictions['rf'] = rf_preds
        
        # Evaluar Gradient Boosting
        if self.gb_model:
            gb_preds = self.evaluate_model(
                self.gb_model, X_test, y_test, 'Gradient Boosting', plot
            )
            predictions['gb'] = gb_preds
        
        # Evaluar Voting Classifier
        if self.voting_model:
            voting_preds = self.evaluate_model(
                self.voting_model, X_test, y_test, 'Voting Classifier', plot
            )
            predictions['voting'] = voting_preds
        
        # Evaluar Red Neuronal
        if self.nn_model:
            nn_preds = self.evaluate_model(
                self.nn_model, X_test, y_test, 'Neural Network', plot
            )
            predictions['nn'] = nn_preds
        
        # Evaluar Ensemble (si al menos dos modelos están disponibles)
        available_models = sum([
            self.rf_model is not None,
            self.gb_model is not None,
            self.voting_model is not None,
            self.nn_model is not None
        ])
        
        if available_models >= 2:
            ensemble_preds = self.ensemble_predict(X_test)
            ensemble_preds_binary = np.where(ensemble_preds > 0.5, 1, 0)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, ensemble_preds_binary)
            precision = precision_score(y_test, ensemble_preds_binary, zero_division=0)
            recall = recall_score(y_test, ensemble_preds_binary, zero_division=0)
            f1 = f1_score(y_test, ensemble_preds_binary, zero_division=0)
            
            # Guardar métricas
            self.model_metrics['Ensemble'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print("\n--- Evaluación del modelo: Ensemble ---")
            print(classification_report(y_test, ensemble_preds_binary))
            
            conf_matrix = confusion_matrix(y_test, ensemble_preds_binary)
            
            if plot:
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis')
                plt.title('Matriz de Confusión - Ensemble', fontsize=16)
                plt.xlabel('Predicciones', fontsize=12)
                plt.ylabel('Valores Verdaderos', fontsize=12)
                plt.tight_layout()
                plt.savefig(f'{self.models_dir}/Ensemble_confusion.png')
                plt.close()
            
            predictions['ensemble'] = ensemble_preds_binary
            
            # Generar gráfico comparativo de métricas
            if plot:
                self._plot_model_comparison()
        
        return predictions
    
    def _plot_model_comparison(self):
        """
        Genera un gráfico comparativo de las métricas de todos los modelos.
        """
        if not self.model_metrics:
            return
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        models = list(self.model_metrics.keys())
        
        # Crear DataFrame para el gráfico
        data = []
        for model in models:
            for metric in metrics:
                data.append({
                    'Modelo': model,
                    'Métrica': metric.capitalize(),
                    'Valor': self.model_metrics[model][metric]
                })
        
        df_metrics = pd.DataFrame(data)
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Modelo', y='Valor', hue='Métrica', data=df_metrics)
        plt.title('Comparación de Modelos', fontsize=16)
        plt.xlabel('Modelo', fontsize=12)
        plt.ylabel('Valor', fontsize=12)
        plt.ylim(0, 1)
        plt.legend(title='Métrica')
        plt.tight_layout()
        plt.savefig(f'{self.models_dir}/model_comparison.png')
        plt.close()
    
    def ensemble_predict(self, X):
        """
        Realiza predicciones combinando todos los modelos.
        
        Args:
            X: Características para predicción
            
        Returns:
            Predicciones ponderadas del ensemble
        """
        predictions = []
        weights = self.ensemble_weights.copy()
        
        # Predicciones de Random Forest
        if self.rf_model:
            rf_pred = self.rf_model.predict_proba(X)[:, 1]
            predictions.append(rf_pred)
        else:
            weights[0] = 0
        
        # Predicciones de Voting Classifier
        if self.voting_model:
            voting_pred = self.voting_model.predict_proba(X)[:, 1]
            predictions.append(voting_pred)
        else:
            weights[1] = 0
        
        # Predicciones de Red Neuronal
        if self.nn_model:
            X_scaled = self.scaler.transform(X)
            nn_pred = self.nn_model.predict(X_scaled).flatten()
            predictions.append(nn_pred)
        else:
            weights[2] = 0
        
        # Predicciones de Gradient Boosting
        if self.gb_model:
            gb_pred = self.gb_model.predict_proba(X)[:, 1]
            predictions.append(gb_pred)
        else:
            weights[3] = 0
        
        # Normalizar pesos si algún modelo falta
        if sum(weights) > 0:
            weights = [w/sum(weights) for w in weights]
        else:
            return None
        
        # Calcular predicción ponderada
        weighted_preds = np.zeros(X.shape[0])
        for i, pred in enumerate(predictions):
            weighted_preds += pred * weights[i]
        
        return weighted_preds
    
    def feature_importance(self, X_test, plot=True):
        """
        Analiza la importancia de características usando SHAP.
        
        Args:
            X_test: Características de prueba
            plot: Si se debe mostrar el gráfico
            
        Returns:
            Valores SHAP
        """
        if self.rf_model is None:
            print("El modelo Random Forest no está disponible para análisis SHAP.")
            return None
        
        print("\nCalculando importancia de características con SHAP...")
        
        # Obtener el mejor estimador
        if hasattr(self.rf_model, 'best_estimator_'):
            model = self.rf_model.best_estimator_
        else:
            model = self.rf_model
        
        # Crear explainer SHAP
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        
        # Mostrar gráfico
        if plot:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, plot_type="bar")
            plt.tight_layout()
            plt.savefig(f'{self.models_dir}/feature_importance.png')
            plt.close()
            
            # Gráfico de dependencia para las características más importantes
            if hasattr(shap_values, 'values'):
                feature_importance = np.abs(shap_values.values).mean(0)
                top_indices = np.argsort(feature_importance)[-3:]  # Top 3 características
                
                for idx in top_indices:
                    if idx < len(self.feature_names):
                        plt.figure(figsize=(10, 6))
                        shap.dependence_plot(idx, shap_values.values, X_test, 
                                            feature_names=self.feature_names)
                        plt.tight_layout()
                        plt.savefig(f'{self.models_dir}/dependence_{self.feature_names[idx]}.png')
                        plt.close()
        
        return shap_values
    
    def predict_next_move(self, board_state, stockfish_path=None):
        """
        Predice el siguiente movimiento basado en el estado del tablero.
        
        Args:
            board_state: Estado del tablero (puede ser un objeto Board o FEN)
            stockfish_path: Ruta al ejecutable de Stockfish (opcional)
            
        Returns:
            Movimiento recomendado y probabilidad
        """
        # Convertir a objeto Board si es una cadena FEN
        if isinstance(board_state, str):
            board = chess.Board(board_state)
        else:
            board = board_state
        
        # Extraer características del tablero
        features = self._extract_board_features(board)
        
        # Convertir a DataFrame con nombres de columnas correctos
        if self.feature_names:
            # Asegurarse de que las características coincidan con las esperadas
            if len(features) != len(self.feature_names):
                print(f"Error: El número de características extraídas ({len(features)}) no coincide con el esperado ({len(self.feature_names)})")
                return None, 0.0
            
            features_df = pd.DataFrame([features], columns=self.feature_names)
        else:
            # Si no tenemos nombres de características, usar un DataFrame simple
            features_df = pd.DataFrame([features])
        
        # Predecir con el ensemble
        win_probability = self.ensemble_predict(features_df)[0]
        
        # Generar movimientos legales
        legal_moves = list(board.legal_moves)
        
        # Si Stockfish está disponible, usarlo para evaluar movimientos
        if stockfish_path:
            try:
                import stockfish
                sf = stockfish.Stockfish(stockfish_path)
                sf.set_fen_position(board.fen())
                best_move = sf.get_best_move()
                return best_move, win_probability
            except Exception as e:
                print(f"Error al usar Stockfish: {e}")
        
        # Si no hay Stockfish o falló, elegir un movimiento basado en heurísticas simples
        if legal_moves:
            # Evaluar cada movimiento
            move_scores = []
            for move in legal_moves:
                # Hacer el movimiento
                board.push(move)
                
                # Extraer características después del movimiento
                next_features = self._extract_board_features(board)
                next_features_df = pd.DataFrame([next_features], columns=self.feature_names) if self.feature_names else pd.DataFrame([next_features])
                
                # Predecir probabilidad de victoria después del movimiento
                next_win_prob = self.ensemble_predict(next_features_df)[0]
                
                # Deshacer el movimiento
                board.pop()
                
                # Guardar puntuación
                move_scores.append((move, next_win_prob))
            
            # Ordenar movimientos por puntuación
            move_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Devolver el mejor movimiento
            best_move = move_scores[0][0]
            return best_move.uci(), win_probability
        
        return None, win_probability
    
    def _extract_board_features(self, board):
        """
        Extrae características de un tablero de ajedrez.
        
        Args:
            board: Objeto Board de python-chess
            
        Returns:
            Vector de características
        """
        features = []
        
        # Material (número de piezas de cada tipo)
        for piece_type in range(1, 7):  # 1=Peón, 2=Caballo, ..., 6=Rey
            features.append(len(board.pieces(piece_type, chess.WHITE)))
            features.append(len(board.pieces(piece_type, chess.BLACK)))
        
        # Control del centro (número de ataques a casillas centrales)
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        white_center_control = sum(len(board.attackers(chess.WHITE, sq)) for sq in center_squares)
        black_center_control = sum(len(board.attackers(chess.BLACK, sq)) for sq in center_squares)
        features.append(white_center_control)
        features.append(black_center_control)
        
        # Movilidad (número de movimientos legales)
        white_mobility = 0
        black_mobility = 0
        
        # Guardar turno actual
        original_turn = board.turn
        
        # Contar movimientos para blancas
        board.turn = chess.WHITE
        white_mobility = board.legal_moves.count()
        
        # Contar movimientos para negras
        board.turn = chess.BLACK
        black_mobility = board.legal_moves.count()
        
        # Restaurar turno original
        board.turn = original_turn
        
        features.append(white_mobility)
        features.append(black_mobility)
        
        # Seguridad del rey (ataques cerca del rey)
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)
        
        if white_king_sq is not None:
            white_king_attackers = len(board.attackers(chess.BLACK, white_king_sq))
            features.append(white_king_attackers)
        else:
            features.append(0)
        
        if black_king_sq is not None:
            black_king_attackers = len(board.attackers(chess.WHITE, black_king_sq))
            features.append(black_king_attackers)
        else:
            features.append(0)
        
        # Estructura de peones
        white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
        
        # Peones doblados (en la misma columna)
        white_pawn_files = [chess.square_file(sq) for sq in white_pawns]
        black_pawn_files = [chess.square_file(sq) for sq in black_pawns]
        
        white_doubled_pawns = len(white_pawn_files) - len(set(white_pawn_files))
        black_doubled_pawns = len(black_pawn_files) - len(set(black_pawn_files))
        
        features.append(white_doubled_pawns)
        features.append(black_doubled_pawns)
        
        # Peones aislados (sin peones en columnas adyacentes)
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
        
        features.append(white_isolated_pawns)
        features.append(black_isolated_pawns)
        
        # Fase de juego (aproximada por número de piezas)
        total_pieces = sum(1 for _ in board.piece_map())
        features.append(total_pieces)
        
        # Turno actual
        features.append(1 if board.turn == chess.WHITE else 0)
        
        # Enroque
        features.append(int(board.has_kingside_castling_rights(chess.WHITE)))
        features.append(int(board.has_queenside_castling_rights(chess.WHITE)))
        features.append(int(board.has_kingside_castling_rights(chess.BLACK)))
        features.append(int(board.has_queenside_castling_rights(chess.BLACK)))
        
        # Jaque
        features.append(int(board.is_check()))
        
        # Desarrollo de piezas (piezas fuera de su posición inicial)
        white_developed = 0
        black_developed = 0
        
        # Posiciones iniciales de piezas menores
        initial_minor_squares = [
            chess.B1, chess.G1, chess.C1, chess.F1,  # Caballos y alfiles blancos
            chess.B8, chess.G8, chess.C8, chess.F8   # Caballos y alfiles negros
        ]
        
        # Contar piezas desarrolladas
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for sq in board.pieces(piece_type, chess.WHITE):
                if sq not in initial_minor_squares:
                    white_developed += 1
            for sq in board.pieces(piece_type, chess.BLACK):
                if sq not in initial_minor_squares:
                    black_developed += 1
        
        features.append(white_developed)
        features.append(black_developed)
        
        return features
    
    def add_performance_to_dataframe(self, df, moves_col='Moves', batch_size=1000):
        """
        Añade columnas de rendimiento al DataFrame basadas en la evaluación de movimientos.
        
        Args:
            df: DataFrame con los datos
            moves_col: Nombre de la columna que contiene los movimientos
            batch_size: Tamaño del lote para procesamiento
            
        Returns:
            DataFrame con columnas de rendimiento añadidas
        """
        print(f"Añadiendo evaluaciones de rendimiento a {len(df)} partidas...")
        
        # Crear copia del DataFrame
        result_df = df.copy()
        
        # Añadir columnas para resultados
        result_df['white_performance'] = 0.0
        result_df['black_performance'] = 0.0
        
        # Procesar en lotes
        for i in range(0, len(result_df), batch_size):
            end_idx = min(i + batch_size, len(result_df))
            print(f"Procesando partidas {i+1} a {end_idx} de {len(result_df)}...")
            
            # Procesar cada partida en el lote
            for idx, row in result_df.iloc[i:end_idx].iterrows():
                try:
                    # Obtener movimientos
                    moves_str = row[moves_col]
                    
                    # Crear tablero
                    board = chess.Board()
                    
                    # Inicializar acumuladores
                    w_total, b_total = 0.0, 0.0
                    move_count = 0
                    
                    # Procesar movimientos
                    for move_str in moves_str.split():
                        try:
                            # Intentar hacer el movimiento
                            move = board.parse_san(move_str)
                            board.push(move)
                            move_count += 1
                            
                            # Cada 4 movimientos, evaluar la posición
                            if move_count % 4 == 0:
                                # Extraer características
                                features = self._extract_board_features(board)
                                features_df = pd.DataFrame([features], columns=self.feature_names) if self.feature_names else pd.DataFrame([features])
                                
                                # Predecir probabilidad de victoria
                                win_prob = self.ensemble_predict(features_df)[0]
                                
                                # Acumular puntuaciones
                                if board.turn == chess.WHITE:  # Último movimiento fue de las negras
                                    b_total += win_prob
                                    w_total += (1 - win_prob)
                                else:  # Último movimiento fue de las blancas
                                    w_total += win_prob
                                    b_total += (1 - win_prob)
                        except Exception as e:
                            # Ignorar errores en movimientos individuales
                            continue
                    
                    # Calcular rendimiento promedio
                    if move_count > 0:
                        evaluations = max(1, move_count // 4)
                        white_perf = (w_total / evaluations) * 100
                        black_perf = (b_total / evaluations) * 100
                        
                        # Guardar resultados
                        result_df.at[idx, 'white_performance'] = white_perf
                        result_df.at[idx, 'black_performance'] = black_perf
                except Exception as e:
                    print(f"Error al procesar partida {idx}: {e}")
        
        print("Procesamiento completado.")
        return result_df

# Ejemplo de uso
if __name__ == "__main__":
    # Crear conjunto de datos de ejemplo (reemplazar con datos reales)
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Crear características aleatorias
    X = np.random.rand(n_samples, n_features)
    
    # Crear etiquetas (0 o 1)
    y = np.random.randint(0, 2, n_samples)
    
    # Crear DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Result'] = y
    
    # Crear y entrenar el ensemble
    ensemble = ChessModelEnsemble(models_dir='Models')
    X_test, y_test = ensemble.train_all_models(df)
    
    # Evaluar modelos
    predictions = ensemble.evaluate_all_models(X_test, y_test)
    
    # Analizar importancia de características
    shap_values = ensemble.feature_importance(X_test)
    
    # Ejemplo de predicción de siguiente movimiento
    board = chess.Board()
    next_move, win_prob = ensemble.predict_next_move(board)
    print(f"\nMovimiento recomendado: {next_move}, Probabilidad de victoria: {win_prob:.2f}")
    
    # Ejemplo de cómo añadir rendimiento a un DataFrame
    # Crear DataFrame de ejemplo con movimientos
    chess_games = pd.DataFrame({
        'Event': ['Ejemplo Partida 1', 'Ejemplo Partida 2'],
        'Result': [1, 0],
        'Moves': ['e4 e5 Nf3 Nc6 Bc4 Nf6', 'd4 d5 c4 e6 Nc3 Nf6 Bg5 Be7']
    })
    
    # Añadir evaluaciones de rendimiento
    games_with_performance = ensemble.add_performance_to_dataframe(chess_games)
    print("\nDataFrame con evaluaciones de rendimiento:")
    print(games_with_performance)
