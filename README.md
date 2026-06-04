# Prediccion de cancelaciones hoteleras

Proyecto final del modulo de Machine Learning y Deep Learning. La propuesta del ejercicio es entrenar un modelo ML para predecir las cancelaciones hoteleras. Nuestro objetivo: Construir un sistema reproducible que entrene, evalue y compare modelos de clasificacion binaria para predecir si una reserva hotelera sera cancelada (`is_canceled = 1`) o no (`is_canceled = 0`).

## Autores y roles

- Victor `<victorh1397@gmail.com>` / `Victorh1397`: analisis exploratorio, limpieza inicial, ingenieria de caracteristicas y primeros experimentos con regresion logistica, arbol de decision y random forest.
- Alejandro Aguado `<alejandro.aguado@gmail.com>`: revision del dataset, identificacion de "data leaks", ampliacion de modelos, API FastAPI y propuesta de dos escenarios de inferencia.

## Datos

El dataset esta en `data/raw/dataset_practica_final.csv` y contiene 119.390 reservas hoteleras con variables de cliente, estancia, canal de venta y comportamiento historico. La variable objetivo es `is_canceled`.

Durante el preprocesamiento general se eliminan columnas administrativas o con fuga de informacion:

- `agent`, `company`
- `reservation_status`, `reservation_status_date`
- `arrival_date_year`

`reservation_status` y `reservation_status_date` se descartan porque contienen informacion posterior al evento que se quiere predecir. En pruebas exploratorias, usar `reservation_status_date` permite alcanzar tasas cercanas al 99.95%, lo que confirma que es una fuga directa de informacion y debe excluirse.

Eliminamos registros con `adr` negativo y filas con nulos en `children` o `country`, siguiendo el analisis de los notebooks.

Para el modelo posterior a la asignacion de habitacion se usa un `feature_set` separado, `post_asignacion`. Este escenario permite usar variables que solo existen despues de asignar habitacion, especialmente `room_changed`, derivada de comparar `reserved_room_type` y `assigned_room_type`. Esta variable no debe usarse en prediccion temprana.

## Estructura

```text
data/
  raw/                         Dataset original
  processed/                   Datasets generados para Victor, Alejandro, binary y post_asignacion
docs/
  informe_final.md             Informe de entrega en Markdown
  checklist_evaluacion.md      Revision contra requisitos de evaluacion
models/
  benchmark_15/                Modelos del benchmark de 15 algoritmos
  finalists_full/              Finalistas entrenados sobre todo el dataset
  optuna_lightgbm_docker_24h/   Mejor modelo pre-asignacion
  feature_set_post_asignacion_victor_params/
                                Mejor modelo post-asignacion
  serving/                      Modelos listos para usar desde FastAPI
  tests/                       Modelos exploratorios previos
notebooks/
  exploracion/                 EDA e investigacion inicial
outputs/
  benchmark_15/                Metricas y graficos del benchmark de 15 algoritmos
  finalists_full/              Metricas de finalistas sobre todo el dataset
  optuna_lightgbm_docker_24h/   Metricas del mejor modelo pre-asignacion
  feature_set_post_asignacion_victor_params/
                                Metricas del mejor modelo post-asignacion
src/
  api.py                       API FastAPI para inferencia con modelos guardados
  data.py                      Carga, limpieza e ingenieria de variables
  models.py                    Registro de modelos y pipelines
  evaluate.py                  Metricas, graficos y guardado de artefactos
  train.py                     CLI del pipeline completo
  tune_optuna.py               Optimizacion de LightGBM con Optuna
tests/                         Pruebas automatizadas
```

## Instalacion

Se recomienda Python 3.11 para maximizar compatibilidad con TensorFlow en Windows.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En entornos donde TensorFlow, XGBoost, LightGBM o CatBoost no esten disponibles, el pipeline sigue siendo ejecutable con alternativas de scikit-learn. En el entorno `.venv` usado para esta entrega quedaron instalados y probados TensorFlow/Keras, XGBoost, LightGBM y CatBoost.

## Ejecucion

El pipeline permite elegir el set de variables:

- `--feature-set victor`: reproduce el dataset modelado por Victor.
- `--feature-set alejandro`: mantiene variables granulares, agrega features de negocio y elimina `assigned_room_type` para reducir riesgo de fuga de informacion.
- `--feature-set binary`: variante experimental con outliers filtrados, cardinalidad reducida y señales binarias de negocio.
- `--feature-set post_asignacion`: modelo posterior a la asignacion de habitacion; puede usar `room_changed`.

Uso recomendado segun el momento de prediccion:

| Momento | Feature set | Modelo recomendado | Motivo |
| --- | --- | --- | --- |
| Antes de asignar habitacion | `victor` | `models/optuna_lightgbm_docker_24h/best_model.joblib` | No usa `room_changed`; valido para prediccion temprana. |
| Despues de asignar habitacion | `post_asignacion` | `models/feature_set_post_asignacion_victor_params/best_model.joblib` | Usa `room_changed`; mas preciso si la habitacion asignada ya existe. |

Generar y entrenar con el dataset Victor:

```powershell
python -m src.train --feature-set victor --models lightgbm --output-dir outputs/feature_set_victor_lightgbm --models-dir models/feature_set_victor_lightgbm --processed-data-path data/processed/dataset_practica_final_modelado_victor.csv --no-plots
```

Generar y entrenar con el dataset de alejandro:

```powershell
python -m src.train --feature-set alejandro --models lightgbm --output-dir outputs/feature_set_alejandro_lightgbm --models-dir models/feature_set_alejandro_lightgbm --processed-data-path data/processed/dataset_practica_final_modelado_alejandro.csv --no-plots
```

Generar y entrenar con el dataset post-asignacion:

```powershell
python -m src.train --feature-set post_asignacion --models lightgbm --output-dir outputs/feature_set_post_asignacion_full_lgbm --models-dir models/feature_set_post_asignacion_full_lgbm --processed-data-path data/processed/dataset_practica_final_modelado_post_asignacion.csv --no-plots
```

Benchmark ampliado con 15 modelos:

```powershell
python -m src.train --feature-set victor --sample-size 10000 --models all --output-dir outputs/benchmark_15 --models-dir models/benchmark_15 --neural-epochs 5
```

Validacion de finalistas sobre todo el dataset:

```powershell
python -m src.train --models lightgbm random_forest catboost bagging gradient_boosting --output-dir outputs/finalists_full --models-dir models/finalists_full --no-plots
```

Tuning de LightGBM con Optuna durante 24 horas en Docker:

```powershell
docker-compose -f docker-compose.optuna.yml up --build -d
docker logs -f hotel-optuna-lightgbm-24h
```

Al finalizar, copiar los resultados del contenedor al proyecto local:

```powershell
docker cp hotel-optuna-lightgbm-8h:/app/outputs/optuna_lightgbm outputs/optuna_lightgbm
docker cp hotel-optuna-lightgbm-8h:/app/models/optuna_lightgbm models/optuna_lightgbm
```

El estudio se guarda de forma incremental en `outputs/optuna_lightgbm/optuna_study.db` dentro del contenedor. Al finalizar genera `metrics.csv`, `best_params.json`, `trials.csv` y `models/optuna_lightgbm/best_model.joblib`.

Ejecucion equivalente sin Docker:

```powershell
python -m src.tune_optuna --feature-set victor --metric f1 --timeout 28800 --cv-splits 3 --output-dir outputs/optuna_lightgbm --models-dir models/optuna_lightgbm --storage sqlite:///outputs/optuna_lightgbm/optuna_study.db --study-name hotel_lightgbm_f1_24h
```

Tuning del modelo post-asignacion:

```powershell
docker-compose -f docker-compose.optuna-post-asignacion.yml up --build -d
docker logs -f hotel-optuna-post-asignacion-24h
```

Levantar API FastAPI para inferencia:

```powershell
$env:HOTEL_MODELS_DIR = "models/serving"
uvicorn src.api:app --reload
```

Modelos disponibles para servir:

| Nombre en API | Feature set que debe usarse | Momento |
| --- | --- | --- |
| `pre_asignacion` | `victor` | Antes de asignar habitacion |
| `post_asignacion` | `post_asignacion` | Despues de asignar habitacion |

Endpoints principales:

- `GET /health`: comprueba que la API responde.
- `GET /models`: lista los modelos `.joblib` disponibles en `HOTEL_MODELS_DIR`.
- `POST /predict/{model_name}`: predice con el modelo seleccionado. El cuerpo debe incluir `feature_set` (`victor`, `alejandro`, `binary` o `post_asignacion`) y `records`, una lista de reservas con el esquema del CSV original sin necesidad de incluir `is_canceled`.

Ejemplo de seleccion de escenario:

```json
{
  "feature_set": "post_asignacion",
  "records": [
    {
      "hotel": "City Hotel",
      "lead_time": 80,
      "arrival_date_year": 2016,
      "arrival_date_month": "December",
      "arrival_date_week_number": 52,
      "arrival_date_day_of_month": 22,
      "stays_in_weekend_nights": 2,
      "stays_in_week_nights": 3,
      "adults": 3,
      "children": 1,
      "babies": 0,
      "meal": "HB",
      "country": "ESP",
      "market_segment": "Online TA",
      "distribution_channel": "TA/TO",
      "is_repeated_guest": 0,
      "previous_cancellations": 1,
      "previous_bookings_not_canceled": 0,
      "reserved_room_type": "D",
      "assigned_room_type": "A",
      "booking_changes": 1,
      "deposit_type": "Non Refund",
      "agent": 9,
      "company": null,
      "days_in_waiting_list": 0,
      "customer_type": "Transient",
      "adr": 110.0,
      "required_car_parking_spaces": 0,
      "total_of_special_requests": 2
    }
  ]
}
```

Ejemplos de endpoint:

```text
POST /predict/pre_asignacion      con "feature_set": "victor"
POST /predict/post_asignacion     con "feature_set": "post_asignacion"
```

Pruebas automatizadas:

```powershell
python -m pytest tests -q
```

## Modelos comparados

- Regresion logistica con class_weight='balanced'
- Arbol de decision con pesos balanceados
- Random Forest con class_weight='balanced_subsample'
- Gradient Boosting con XGBoost
- Red neuronal multicapa con TensorFlow/Keras
- Extra Trees
- AdaBoost
- Bagging
- K-Nearest Neighbors
- SVM con kernel RBF
- Linear SVM
- Gaussian Naive Bayes
- Ridge Classifier
- LightGBM
- CatBoost

La metrica principal seleccionada es F1-score porque el problema tiene desbalance moderado entre reservas no canceladas y canceladas. F1 equilibra precision y recall, lo que evita elegir un modelo que parezca bueno por accuracy pero pierda demasiadas cancelaciones reales.

## Resultados verificados

Modelos finales preparados:

| Escenario | Feature set | Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Antes de asignar habitacion | `victor` | `models/optuna_lightgbm_docker_24h/best_model.joblib` | 0.8849 | 0.8353 | 0.8596 | 0.8473 | 0.9559 |
| Despues de asignar habitacion | `post_asignacion` | `models/feature_set_post_asignacion_victor_params/best_model.joblib` | 0.8888 | 0.8418 | 0.8625 | 0.8521 | 0.9586 |

La mejora del modelo post-asignacion depende principalmente de `room_changed`. Por eso no sustituye al modelo pre-asignacion: son dos modelos para dos momentos distintos del proceso.

Optuna adicional de 24 horas sobre `post_asignacion`:

| Modelo | Trials | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `models/optuna_post_asignacion_lgbm_8h/best_model.joblib` | 661 | 0.8882 | 0.8410 | 0.8620 | 0.8514 | 0.9584 |

Este Optuna confirma la mejora del escenario posterior, aunque no supera al modelo post-asignacion generado con los parametros Optuna de Victor (`F1=0.8521`).

Comparacion de feature sets con LightGBM sobre todo el dataset:

| Feature set | Filas | Columnas | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Victor | 118.897 | 27 | 0.855 | 0.780 | 0.849 | 0.813 | 0.939 |
| Alejandro | 118.897 | 39 | 0.856 | 0.783 | 0.846 | 0.813 | 0.938 |

La version Alejandro mejora ligeramente F1 (`0.81325` frente a `0.81282`) y evita `assigned_room_type`. La version Victor conserva un ROC-AUC ligeramente superior.

Benchmark ampliado local con el feature set Victor: `--sample-size 10000`, 15 modelos, metrica principal `f1`.

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.851 | 0.788 | 0.818 | 0.803 | 0.922 |
| Random Forest | 0.844 | 0.796 | 0.779 | 0.788 | 0.916 |
| CatBoost | 0.848 | 0.826 | 0.747 | 0.784 | 0.918 |
| Bagging | 0.840 | 0.784 | 0.785 | 0.784 | 0.912 |
| XGBoost | 0.847 | 0.827 | 0.744 | 0.783 | 0.917 |
| SVM RBF | 0.830 | 0.756 | 0.799 | 0.777 | 0.901 |
| Extra Trees | 0.831 | 0.765 | 0.785 | 0.775 | 0.910 |
| Arbol de decision | 0.804 | 0.708 | 0.802 | 0.752 | 0.889 |
| Regresion logistica | 0.792 | 0.700 | 0.767 | 0.732 | 0.882 |
| Ridge Classifier | 0.792 | 0.703 | 0.759 | 0.730 | 0.877 |
| Linear SVM | 0.785 | 0.692 | 0.756 | 0.723 | 0.881 |
| Red neuronal Keras | 0.808 | 0.782 | 0.670 | 0.722 | 0.880 |
| KNN | 0.810 | 0.794 | 0.659 | 0.721 | 0.879 |
| AdaBoost | 0.802 | 0.864 | 0.555 | 0.675 | 0.890 |
| GaussianNB | 0.546 | 0.445 | 0.896 | 0.594 | 0.788 |

Validacion de los 5 finalistas sobre todo el dataset:

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.855 | 0.780 | 0.849 | 0.813 | 0.939 |
| Random Forest | 0.858 | 0.802 | 0.819 | 0.810 | 0.936 |
| CatBoost | 0.856 | 0.841 | 0.756 | 0.796 | 0.932 |
| Bagging | 0.846 | 0.785 | 0.807 | 0.796 | 0.930 |
| XGBoost | 0.852 | 0.831 | 0.756 | 0.792 | 0.930 |

El modelo seleccionado por F1 fue LightGBM, guardado en `models/finalists_full/best_model.joblib`.

## Artefactos

La ejecucion del pipeline genera:

- `outputs/benchmark_15/metrics.csv`
- `outputs/feature_set_victor_lightgbm/metrics.csv`
- `outputs/feature_set_alejandro_lightgbm/metrics.csv`
- `outputs/benchmark_15/best_model_summary.json`
- `outputs/benchmark_15/confusion_matrix_<modelo>.png`
- `outputs/benchmark_15/roc_curve_<modelo>.png`
- `outputs/benchmark_15/roc_curves_comparison.png`
- `outputs/benchmark_15/feature_importance_<modelo>.png` cuando el modelo expone importancias o coeficientes
- `outputs/finalists_full/metrics.csv`
- `outputs/optuna_lightgbm/metrics.csv`, `best_params.json`, `trials.csv` y `optuna_study.db`
- `outputs/optuna_lightgbm_docker_24h/metrics.csv`, `best_params.json`, `trials.csv` y `optuna_study.db`
- `outputs/feature_set_post_asignacion_victor_params/metrics.csv`
- `outputs/optuna_post_asignacion_lgbm_8h/metrics.csv`, `best_params.json`, `trials.csv` y `optuna_study.db`
- `data/processed/dataset_practica_final_modelado_victor.csv`
- `data/processed/dataset_practica_final_modelado_alejandro.csv`
- `data/processed/dataset_practica_final_modelado_binary.csv`
- `data/processed/dataset_practica_final_modelado_post_asignacion.csv`
- `models/benchmark_15/<modelo>.joblib`
- `models/finalists_full/best_model.joblib`
- `models/optuna_lightgbm/best_model.joblib`
- `models/optuna_lightgbm_docker_24h/best_model.joblib`
- `models/feature_set_post_asignacion_victor_params/best_model.joblib`
- `models/optuna_post_asignacion_lgbm_8h/best_model.joblib`
- `models/serving/pre_asignacion.joblib`
- `models/serving/post_asignacion.joblib`
- `src/api.py` con `GET /models` y `POST /predict/{model_name}`

## Conclusiones

Tras instalar y probar XGBoost, LightGBM, CatBoost y TensorFlow/Keras, el mejor algoritmo para este problema es LightGBM.

El proyecto queda con dos modelos finales:

- Modelo pre-asignacion: usa `feature_set=victor`, no depende de `assigned_room_type` ni de `room_changed`, y alcanza F1 `0.8473`.
- Modelo post-asignacion: usa `feature_set=post_asignacion`, incorpora `room_changed` cuando la habitacion asignada ya existe, y alcanza F1 `0.8521`.

La version post-asignacion es mas precisa, pero solo debe usarse cuando la prediccion se haga despues de asignar habitacion. Para prediccion temprana, el modelo correcto es el pre-asignacion.
