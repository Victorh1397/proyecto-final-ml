# Prediccion de cancelaciones hoteleras

Proyecto final del modulo de Machine Learning y Deep Learning. El objetivo es construir un sistema reproducible que entrene, evalue y compare modelos de clasificacion binaria para predecir si una reserva hotelera sera cancelada (`is_canceled = 1`) o no (`is_canceled = 0`).

## Autores y roles

- Victor `<victorh1397@gmail.com>` / `Victorh1397`: analisis exploratorio, limpieza inicial, ingenieria de caracteristicas y primeros experimentos con regresion logistica, arbol de decision y random forest.
- Alejandro Aguado `<alejandro.aguado@gmail.com>`: revision del dataset, identificacion de "data leaks" y propuesta de mejoras alternativas poco eticas.

## Datos

El dataset esta en `data/raw/dataset_practica_final.csv` y contiene 119.390 reservas hoteleras con variables de cliente, estancia, canal de venta y comportamiento historico. La variable objetivo es `is_canceled`.

Durante el preprocesamiento se eliminan columnas administrativas o con fuga de informacion:

- `agent`, `company`
- `reservation_status`, `reservation_status_date`
- `arrival_date_year`

`reservation_status` y `reservation_status_date` se descartan porque contienen informacion posterior al evento que se quiere predecir. En pruebas para ganar la taza , usar `reservation_status_date` permite alcanzar tasas cercanas al 99.95% y aunque en las normas del proyecto no hay ninguna referencia explicita a no usar "data leaks", Victor no se sentia moralmente comodo y lo hemos excluido. 

Eliminamos registros con `adr` negativo y filas con nulos en `children` o `country`, siguiendo el analisis de los notebooks.

## Estructura

```text
data/
  raw/                         Dataset original
  processed/                   Datasets generados para Victor y Alejandro
docs/
  informe_final.md             Informe de entrega en Markdown
  checklist_evaluacion.md      Revision contra requisitos de evaluacion
models/
  benchmark_15/                Modelos del benchmark de 15 algoritmos
  finalists_full/              Finalistas entrenados sobre todo el dataset
  tests/                       Modelos exploratorios previos
notebooks/
  exploracion/                 EDA e investigacion inicial
outputs/
  benchmark_15/                Metricas y graficos del benchmark de 15 algoritmos
  finalists_full/              Metricas de finalistas sobre todo el dataset
src/
  api.py                       API FastAPI para inferencia con modelos guardados
  data.py                      Carga, limpieza e ingenieria de variables
  models.py                    Registro de modelos y pipelines
  evaluate.py                  Metricas, graficos y guardado de artefactos
  train.py                     CLI del pipeline completo
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

Generar y entrenar con el dataset Victor:

```powershell
python -m src.train --feature-set victor --models lightgbm --output-dir outputs/feature_set_victor_lightgbm --models-dir models/feature_set_victor_lightgbm --processed-data-path data/processed/dataset_practica_final_modelado_victor.csv --no-plots
```

Generar y entrenar con el dataset de alejandro:

```powershell
python -m src.train --feature-set alejandro --models lightgbm --output-dir outputs/feature_set_alejandro_lightgbm --models-dir models/feature_set_alejandro_lightgbm --processed-data-path data/processed/dataset_practica_final_modelado_alejandro.csv --no-plots
```

Benchmark ampliado con 15 modelos:

```powershell
python -m src.train --feature-set victor --sample-size 10000 --models all --output-dir outputs/benchmark_15 --models-dir models/benchmark_15 --neural-epochs 5
```

Validacion de finalistas sobre todo el dataset:

```powershell
python -m src.train --models lightgbm random_forest catboost bagging gradient_boosting --output-dir outputs/finalists_full --models-dir models/finalists_full --no-plots
```

Levantar API FastAPI para inferencia:

```powershell
$env:HOTEL_MODELS_DIR = "models/finalists_full"
uvicorn src.api:app --reload
```

Endpoints principales:

- `GET /health`: comprueba que la API responde.
- `GET /models`: lista los modelos `.joblib` disponibles en `HOTEL_MODELS_DIR`.
- `POST /predict/{model_name}`: predice con el modelo seleccionado. El cuerpo debe incluir `feature_set` (`victor` o `alejandro`) y `records`, una lista de reservas con el esquema del CSV original sin necesidad de incluir `is_canceled`.

Pruebas automatizadas:

```powershell
python -m pytest tests -q
```

## Modelos comparados

- Regresion logistica con `class_weight='balanced'`
- Arbol de decision con pesos balanceados
- Random Forest con `class_weight='balanced_subsample'`
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
- `data/processed/dataset_practica_final_modelado_victor.csv`
- `data/processed/dataset_practica_final_modelado_alejandro.csv`
- `models/benchmark_15/<modelo>.joblib`
- `models/finalists_full/best_model.joblib`
- `src/api.py` con `GET /models` y `POST /predict/{model_name}`

## Conclusiones

Tras instalar y probar XGBoost, LightGBM, CatBoost y TensorFlow/Keras, el mejor resultado lo obtiene LightGBM. En dataset completo, LightGBM alcanza F1 `0.813` y ROC-AUC `0.939`, mejorando el Random Forest previo. La version `alejandro` queda disponible como alternativa porque evita usar la habitacion asignada aunque el resultado es algo peor.

