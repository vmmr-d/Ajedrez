import streamlit as st
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    st.warning("Matplotlib and Seaborn are not installed. Visualization features will be disabled.")
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

    def __init__(self, models_dir='models/chess_predictor'):
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

    def prepare_data_from_dataframe(self, df, moves_col='Moves', result_col='Result', elo_cols=None,
                                    exclude_cols=None):
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
        if HAS_MATPLOTLIB:
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
            Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(2)  # Salida para predecir dos valores (rendimiento de blancas y negras)
        ])

        # Compilar el modelo
        nn_model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

        # Callbacks para mejorar el entrenamiento
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        model_checkpoint = ModelCheckpoint(f'{self.models_dir}/performance_nn_model.h5', save_best_only=True)

        # Entrenar el modelo
        history = nn_model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=0
        )

        # Cargar el mejor modelo guardado
        nn_model = load_model(f'{self.models_dir}/performance_nn_model.h5')
        self.performance_model = nn_model

        # Evaluar el modelo
        loss = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"Pérdida del modelo de red neuronal: {loss:.4f}")

        # Hacer predicciones
        y_pred = nn_model.predict(X_test_scaled)

        # Calcular métricas de evaluación
        r2_white = r2_score(y_test[:, 0], y_pred[:, 0])
        r2_black = r2_score(y_test[:, 1], y_pred[:, 1])
        print(f"R2 para rendimiento de blancas: {r2_white:.4f}")
        print(f"R2 para rendimiento de negras: {r2_black:.4f}")

        return X_test_scaled, y_test

    def _train_single_performance_model(self, X_train, y_train, X_test, y_test, color, batch_size=10000):
        """
        Entrena un modelo XGBoost para predecir el rendimiento de un solo color.

        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de rendimiento de entrenamiento
            X_test: Características de prueba
            y_test: Etiquetas de rendimiento de prueba
            color: 'white' o 'black' para identificar el modelo
            batch_size: Tamaño del lote para entrenamiento

        Returns:
            Modelo entrenado
        """
        print(f"Entrenando modelo para rendimiento de {color}...")

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
        base_model = XGBRegressor(
            objective='reg:squarederror',  # Cambiado a regresión
            random_state=42,
            tree_method='hist',
            enable_categorical=False
        )

        # Realizar búsqueda de hiperparámetros
        print("Realizando búsqueda de hiperparámetros (esto puede tomar tiempo)...")

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
            scoring='r2',
            cv=3,
            random_state=42,
            n_jobs=-1
        )

        search.fit(X_sample, y_sample)

        print(f"Mejores hiperparámetros: {search.best_params_}")

        # Entrenar modelo final con los mejores parámetros
        model = XGBRegressor(
            objective='reg:squarederror',  # Cambiado a regresión
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
                eval_set=[(X_test, y_test)],  # Conjunto de evaluación
                verbose=False
            )

            # Continuar entrenando con el resto de lotes
            for start_idx in tqdm(range(first_batch_size, X_train.shape[0], batch_size), desc="Entrenando lotes"):
                end_idx = min(start_idx + batch_size, X_train.shape[0])
                model.fit(
                    X_train[start_idx:end_idx],
                    y_train[start_idx:end_idx],
                    xgb_model=model,  # Continuar desde el modelo actual
                    eval_set=[(X_test, y_test)],
                    verbose=False
                )
        else:
            # Entrenar con todos los datos a la vez
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

        # Guardar modelo
        joblib.dump(model, f'{self.models_dir}/{color}_performance_model.pkl')

        # Evaluar modelo
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"R2 para rendimiento de {color}: {r2:.4f}")

        return model

    def predict_result(self, features):
        """
        Predice el resultado de una partida a partir de sus características.

        Args:
            features: Vector de características.

        Returns:
            Resultado predicho (0 Gana Blanca, 1 Gana Negra, 0.5 Empate).
        """
        if self.result_model is None:
            print("Modelo de resultado no entrenado.")
            return None

        # Asegurarse de que las características son un array 2D
        features = np.array(features).reshape(1, -1)

        # Escalar características
        features_scaled = self.scaler.transform(features)

        # Predecir resultado
        prediction = self.result_model.predict(features_scaled)[0]

        return prediction

    def predict_performance(self, features):
        """
        Predice el rendimiento de una partida a partir de sus características.

        Args:
            features: Vector de características.

        Returns:
            Rendimiento predicho para blancas y negras.
        """
        if self.performance_model is None:
            print("Modelo de rendimiento no entrenado.")
            return None

        # Asegurarse de que las características son un array 2D
        features = np.array(features).reshape(1, -1)

        # Escalar características
        features_scaled = self.scaler.transform(features)

        # Predecir rendimiento
        prediction = self.performance_model.predict(features_scaled)[0]

        return prediction


# Configuración de la página
st.set_page_config(page_title="Predictor de Ajedrez", layout="wide")

# Título de la aplicación
st.title("Predictor de Resultados y Rendimiento en Ajedrez")

# Sidebar para seleccionar la acción
action = st.sidebar.selectbox(
    "Seleccione una acción",
    ["Entrenar modelo", "Predecir resultado", "Analizar partida"]
)

# Inicializar el predictor
predictor = ChessResultPredictor()

if action == "Entrenar modelo":
    st.header("Entrenamiento del Modelo")

    # Subir archivo CSV
    uploaded_file = st.file_uploader("Cargar archivo CSV con partidas", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Datos cargados correctamente. Muestra de los datos:")
        st.dataframe(df.head())

        if st.button("Entrenar modelo"):
            with st.spinner("Entrenando modelo..."):
                X, y_result, y_performance = predictor.prepare_data_from_dataframe(df)
                predictor.train_result_model(X, y_result)
                predictor.train_performance_model(X, y_performance)
            st.success("Modelo entrenado exitosamente!")

            # Mostrar gráficos de importancia de características
            st.subheader("Importancia de Características")
            if HAS_MATPLOTLIB:
                try:
                    st.image('models/chess_predictor/result_feature_importance.png')
                except FileNotFoundError:
                    st.warning("El gráfico de importancia de características no se pudo generar. Asegúrese de que el modelo se haya entrenado correctamente.")
            else:
                st.warning("Matplotlib is not installed. Feature importance graph cannot be displayed.")


elif action == "Predecir resultado":
    st.header("Predicción de Resultado")

    # Entrada de movimientos
    moves = st.text_area("Ingrese los movimientos de la partida (notación algebraica)")

    if st.button("Predecir"):
        if predictor.result_model is None:
            st.error("El modelo no ha sido entrenado. Por favor, entrene el modelo primero.")
        else:
            features, _ = predictor.extract_features_from_game(moves)
            result = predictor.predict_result(features)
            st.write(f"Resultado predicho: {result}")

elif action == "Analizar partida":
    st.header("Análisis de Partida")

    # Entrada de movimientos
    moves = st.text_area("Ingrese los movimientos de la partida (notación algebraica)")

    if st.button("Analizar"):
        if predictor.result_model is None or predictor.performance_model is None:
            st.error("Los modelos no han sido entrenados. Por favor, entrene los modelos primero.")
        else:
            features, feature_names = predictor.extract_features_from_game(moves)
            result = predictor.predict_result(features)
            performance = predictor.predict_performance(features)

            st.write(f"Resultado predicho: {result}")
            st.write(f"Rendimiento predicho - Blancas: {performance[0]:.2f}%, Negras: {performance[1]:.2f}%")

            # Mostrar características más importantes
            st.subheader("Características más importantes")
            feature_importance = predictor.feature_importance
            if feature_importance:
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                for feature, importance in sorted_features:
                    st.write(f"{feature}: {importance:.4f}")

# Pie de página
st.sidebar.markdown("---")
st.sidebar.text("Desarrollado por Victor, Medrano Jesus Rodriguez, Enrique Esnaola")