# Sistema de Análisis y Predicción de Partidas de Ajedrez

Este proyecto implementa un sistema completo para analizar partidas de ajedrez, extraer características relevantes, calcular métricas de rendimiento y predecir resultados utilizando técnicas de aprendizaje automático.

## Contenido

- [Descripción General](#descripción-general)
- [Componentes Principales](#componentes-principales)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
  - [Extracción de Características](#extracción-de-características)
  - [Entrenamiento de Modelos](#entrenamiento-de-modelos)
  - [Predicción y Análisis](#predicción-y-análisis)
- [Ejemplos](#ejemplos)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Características Extraídas](#características-extraídas)
- [Modelos Implementados](#modelos-implementados)

## Descripción General

Este sistema permite analizar partidas de ajedrez a partir de su notación algebraica, extrayendo características relevantes como control del centro, movilidad, seguridad del rey, estructura de peones, etc. Estas características se utilizan para:

1. Calcular métricas de rendimiento para ambos jugadores
2. Entrenar modelos de aprendizaje automático para predecir resultados
3. Generar análisis detallados de partidas

El sistema está optimizado para manejar grandes conjuntos de datos (más de 120,000 partidas) y utiliza técnicas avanzadas de procesamiento por lotes y paralelización.

## Componentes Principales

El proyecto consta de dos componentes principales:

1. **Evaluador**: Clase encargada de extraer características de partidas de ajedrez y calcular métricas de rendimiento.
2. **ChessResultPredictor**: Clase que implementa modelos de aprendizaje automático para predecir resultados y rendimiento de partidas.

## Requisitos

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- chess
- scikit-learn
- xgboost
- tensorflow
- joblib
- tqdm

## Instalación

1. Clona este repositorio:
```bash
git clone https://github.com/vmmr-d/Ajedrez.git
cd chess-analysis-system
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Extracción de Características

Para extraer características de partidas de ajedrez y calcular métricas de rendimiento:

```python
import pandas as pd
from evaluador import Evaluador

# Cargar datos
df = pd.read_csv('partidas_ajedrez.csv')

# Crear instancia del evaluador
evaluador = Evaluador()

# Extraer características y calcular rendimiento
df_con_caracteristicas = evaluador.calculate_performance_labels(
    df=df,
    moves_column='Moves',  # Columna que contiene los movimientos
    n_jobs=-1,             # Usar todos los núcleos disponibles
    batch_size=5000        # Tamaño del lote para procesamiento
)

# Guardar resultados
df_con_caracteristicas.to_csv('partidas_con_caracteristicas.csv', index=False)
```

### Entrenamiento de Modelos

Para entrenar modelos de predicción de resultados y rendimiento:

```python
from Ajedrez_Prediccion import ChessResultPredictor

# Crear instancia del predictor
predictor = ChessResultPredictor(models_dir='../models/mi_modelo_ajedrez')

# Entrenar modelos
eval_results = predictor.train_models(
    df=df_con_caracteristicas,
    moves_col='Moves',
    result_col='Result',
    batch_size=10000  # Ajustar según capacidad de memoria
)

print("Modelos entrenados y guardados en: ../models/mi_modelo_ajedrez")
```

### Predicción y Análisis

Para analizar una partida y predecir su resultado:

```python
# Cargar un predictor previamente entrenado
predictor = ChessResultPredictor(models_dir='../models/mi_modelo_ajedrez')
predictor.load_models()

# Partida a analizar (notación algebraica)
partida = 'e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O'

# Generar informe detallado
informe = predictor.generate_game_report(
    moves_str=partida,
    output_file='informe_partida.txt',
    white_elo=1850,
    black_elo=1750
)

print("Informe generado y guardado en: informe_partida.txt")
```

## Ejemplos

### Ejemplo 1: Extraer características de una partida individual

```python
from evaluador import Evaluador

evaluador = Evaluador()
partida = 'e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O'
features, feature_names = evaluador.extract_features_from_game(partida)

# Imprimir características extraídas
for name, value in zip(feature_names, features):
    print(f"{name}: {value}")
```

### Ejemplo 2: Predecir resultado para múltiples partidas

```python
import pandas as pd
from Ajedrez_Prediccion import ChessResultPredictor

# Cargar datos
df = pd.read_csv('nuevas_partidas.csv')

# Cargar predictor
predictor = ChessResultPredictor(models_dir='../models/mi_modelo_ajedrez')
predictor.load_models()

# Predecir resultados
df_con_predicciones = predictor.predict_dataframe(
    df=df,
    moves_col='Moves',
    batch_size=1000
)

# Guardar resultados
df_con_predicciones.to_csv('partidas_con_predicciones.csv', index=False)
```

## Estructura del Proyecto

```
chess-analysis-system/
├── evaluador.py  # Extractor de características
├── Ajedrez_Prediccion.py  # Predictor de resultados
├── requirements.txt  # Dependencias del proyecto
├── examples/  # Ejemplos de uso
│   ├── extract_features.py
│   ├── train_models.py
│   └── analyze_game.py
├── models/  # Directorio para guardar modelos entrenados
└── data/  # Datos de ejemplo
    └── sample_games.csv
```

## Características Extraídas

El sistema extrae y calcula más de 30 características de cada partida, incluyendo:

- **Material**: Cantidad promedio de piezas de cada tipo para ambos jugadores
- **Control del centro**: Dominio de las casillas centrales
- **Movilidad**: Cantidad de movimientos legales disponibles
- **Seguridad del rey**: Defensas y ataques alrededor del rey
- **Estructura de peones**: Peones doblados, aislados, etc.
- **Desarrollo**: Tiempo de desarrollo de piezas menores
- **Actividad**: Control del tablero y ataques a piezas
- **Estadísticas**: Capturas, jaques, movimientos totales, etc.

## Modelos Implementados

El sistema implementa varios modelos de aprendizaje automático:

1. **Modelo de clasificación XGBoost**: Para predecir el resultado de la partida (victoria blancas, tablas, victoria negras)
2. **Modelos de regresión XGBoost**: Para predecir el rendimiento individual de cada jugador
3. **Red neuronal**: Modelo de aprendizaje profundo para predecir simultáneamente el rendimiento de ambos jugadores

Los modelos se entrenan con búsqueda de hiperparámetros y validación cruzada para obtener el mejor rendimiento posible.
