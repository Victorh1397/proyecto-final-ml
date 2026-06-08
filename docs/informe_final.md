# Informe final - Prediccion de cancelaciones hoteleras

## 1. Participantes

- Victor "victorh1397@gmail.com"
- Alejandro Aguado "alejandro.aguado@gmail.com"

## 2. Analisis inicial

Principales datos:

- El valor objetivo tiene desbalance moderado: aproximadamente 63% reservas no canceladas y 37% canceladas.
- `lead_time` muestra una relacion importante con la cancelacion: las reservas hechas con mas antelacion tienden a cancelarse mas.
- `reservation_status` y `reservation_status_date` se eliminaron por ser un "data leak" , ya que representan informacion posterior al estado final de la reserva.
- `agent` y `company` se eliminan del modelo pre-asignacion por ser identificadores administrativos con muchos nulos. En el modelo post-asignacion solo se transforman en flags (`has_agent`, `has_company`) y despues se eliminan como identificadores brutos.
- Se detecto un valor negativo en `adr`, que se descarta por incoherencia.
- Los nulos en `children` y `country` son porcentualmente bajos, por lo que se eliminan.

## 3. Decisiones sobre dataset

Se mantienen varias versiones de modelado para poder comparar resultados sin pisar entregables previos.

Version Victor:

- `adults` -> `adults_categories`: 1 adulto, 2 adultos, 3 o mas adultos.
- `children` -> `has_children`.
- `babies` -> `has_babies`.
- `arrival_date_day_of_month` -> `month_period`.
- `arrival_date_week_number` -> `arrival_season`.
- `arrival_date_month` -> `arrival_quarter`.
- `country` -> `country_grouped`, conservando los 15 paises mas frecuentes y agrupando el resto.

Esta version elimina las columnas originales reemplazadas y conserva `assigned_room_type`, replicando el dataset trabajado previamente.

Version Alejandro:

- Mantiene `adults`, `children`, `babies`, `arrival_date_month`, `arrival_date_week_number`, `arrival_date_day_of_month` y `country`.
- Agrega las variables derivadas de Victor.
- Agrega `total_nights`, `total_guests`, `adr_per_guest`, `has_special_requests`, `has_parking_request` y `lead_time_bucket`.
- Elimina `assigned_room_type` para reducir la posibilidad de "data leak" si la prediccion se hace antes de asignar habitacion.

Version post-asignacion:

- Se usa cuando la habitacion ya ha sido asignada.
- Agrega `total_guests`, `total_nights`, `adr_per_person`, `previous_cancel_ratio`, `room_changed`, `has_agent`, `has_company` y `arrival_season`.
- `room_changed` se calcula comparando `reserved_room_type` y `assigned_room_type`.
- Despues de crear las variables derivadas se eliminan `agent`, `company` y `assigned_room_type` como columnas brutas.

Decision final:

- Si la prediccion se hace antes de asignar habitacion, no se puede usar `room_changed`.
- Si la prediccion se hace despues de asignar habitacion, `room_changed` si es defendible porque la informacion ya existe en el proceso.

Por este motivo el proyecto termina con dos modelos finales: uno pre-asignacion y otro post-asignacion.

## 4. Diseño del sistema

El proyecto se organiza como un pipeline ejecutable:

```text
src/data.py      -> carga, limpieza, feature engineering y split X/y
src/models.py    -> definicion de modelos y preprocesador comun
src/evaluate.py  -> metricas, graficos y artefactos
src/train.py     -> CLI que orquesta el flujo completo
src/tune_optuna.py -> optimizacion de LightGBM con Optuna
src/api.py       -> API FastAPI para inferencia con modelos guardados
tests/           -> pruebas de limpieza, features, entrenamiento, API y backend grafico
```

El preprocesador usa `ColumnTransformer` con:

- `OneHotEncoder(handle_unknown='ignore', drop='first')` para categoricas.
- `StandardScaler` para numericas.

La seleccion del modelo se realiza segun F1-score, manteniendo tambien accuracy, precision, recall y ROC-AUC para comparacion.

El CLI permite seleccionar la version con `--feature-set victor`, `--feature-set alejandro`, `--feature-set binary` o `--feature-set post_asignacion`. Tambien permite guardar el CSV usado con `--processed-data-path`.

Para optimizacion de hiperparametros se incorpora `src/tune_optuna.py`, que ejecuta Optuna sobre LightGBM con validacion cruzada, guarda el estudio SQLite para reanudar ejecuciones y exporta `best_params.json`, `trials.csv`, `metrics.csv` y el mejor modelo. Tambien se incluye `docker-compose.optuna.yml` para dejar una busqueda de 24 horas en un servidor Docker con mas recursos:

```powershell
docker-compose -f docker-compose.optuna.yml up --build -d
docker logs -f hotel-optuna-lightgbm-24h
docker cp hotel-optuna-lightgbm-24h:/app/outputs/optuna_lightgbm outputs/optuna_lightgbm
docker cp hotel-optuna-lightgbm-24h:/app/models/optuna_lightgbm models/optuna_lightgbm
```

Como mejora tecnica adicional, se incluye una API FastAPI para servir inferencia:

- `GET /health`: validacion basica de servicio.
- `GET /models`: listado de modelos `.joblib` disponibles.
- `POST /predict/{model_name}`: prediccion con el modelo seleccionado y el feature set indicado.

Se puede levantar con:

```powershell
$env:HOTEL_MODELS_DIR = "models/serving"
uvicorn src.api:app --reload
```

Modelos preparados para servir:

| Modelo API | Feature set | Momento de uso |
| --- | --- | --- |
| `pre_asignacion` | `victor` | Antes de asignar habitacion |
| `post_asignacion` | `post_asignacion` | Despues de asignar habitacion |

## 5. Modelos implementados

Modelos disponibles en el pipeline:

- Regresion logistica.
- Arbol de decision.
- Random Forest.
- Gradient Boosting con XGBoost.
- Red neuronal multicapa con TensorFlow/Keras.
- Extra Trees.
- AdaBoost.
- Bagging.
- K-Nearest Neighbors.
- SVM con kernel RBF.
- Linear SVM.
- Gaussian Naive Bayes.
- Ridge Classifier.
- LightGBM.
- CatBoost.

Ademas, se incluye balanceo con `class_weight` en los modelos que lo soportan, opcion `--tune` para usar `GridSearchCV` en modelos scikit-learn y tuning avanzado con Optuna para LightGBM.

## 6. Metrica principal

La metrica principal es F1-score. La razon es que el dataset no esta perfectamente balanceado y no se concentra solo en una clase:

- Un falso negativo implica no detectar una reserva que se cancelara.
- Un falso positivo puede activar acciones comerciales innecesarias.

F1 equilibra precision y recall, y por tanto es mas informativa que accuracy para tomar esta decision.

## 7. Resultados

Modelos finales preparados:

| Escenario | Feature set | Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Antes de asignar habitacion | `victor` | `models/serving/pre_asignacion.joblib` | 0.8849 | 0.8353 | 0.8596 | 0.8473 | 0.9559 |
| Despues de asignar habitacion | `post_asignacion` | `models/serving/post_asignacion.joblib` | 0.8888 | 0.8418 | 0.8625 | 0.8521 | 0.9586 |

La diferencia entre ambos modelos no es solo tecnica, sino operativa. El modelo pre-asignacion se puede usar antes de conocer la habitacion asignada. El modelo post-asignacion es mas preciso, pero solo debe usarse cuando `assigned_room_type` ya existe y por tanto puede calcularse `room_changed`.

Resultado de Optuna sobre el modelo pre-asignacion:

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM Optuna pre-asignacion | 0.8849 | 0.8353 | 0.8596 | 0.8473 | 0.9559 |

Resultado de Optuna 24h sobre el modelo post-asignacion:

| Modelo | Trials | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM Optuna post-asignacion | 661 | 0.8882 | 0.8410 | 0.8620 | 0.8514 | 0.9584 |

El mejor resultado post-asignacion no fue el Optuna de 24 horas, sino el modelo que reutiliza los mejores parametros Optuna encontrados en Victor sobre el feature set `post_asignacion`: F1 `0.8521`.

Comparacion de versiones con LightGBM sobre todo el dataset:

```powershell
python -m src.train --feature-set victor --models lightgbm --output-dir outputs/feature_set_victor_lightgbm --models-dir models/feature_set_victor_lightgbm --processed-data-path data/processed/dataset_practica_final_modelado_victor.csv --no-plots

python -m src.train --feature-set alejandro --models lightgbm --output-dir outputs/feature_set_alejandro_lightgbm --models-dir models/feature_set_alejandro_lightgbm --processed-data-path data/processed/dataset_practica_final_modelado_alejandro.csv --no-plots
```

| Feature set | Filas | Columnas | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Victor | 118.897 | 27 | 0.855 | 0.780 | 0.849 | 0.813 | 0.939 |
| Alejandro | 118.897 | 39 | 0.856 | 0.783 | 0.846 | 0.813 | 0.938 |

La version Alejandro mejora ligeramente F1 (`0.81325` frente a `0.81282`) y evita `assigned_room_type`. La version Victor conserva un ROC-AUC ligeramente superior.

Benchmark local ampliado con 15 modelos:

```powershell
python -m src.train --feature-set victor --sample-size 10000 --models all --output-dir outputs/benchmark_15 --models-dir models/benchmark_15 --neural-epochs 5
```

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

Despues se validaron los 5 finalistas sobre todo el dataset:

```powershell
python -m src.train --models lightgbm random_forest catboost bagging gradient_boosting --output-dir outputs/finalists_full --models-dir models/finalists_full --no-plots
```

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.855 | 0.780 | 0.849 | 0.813 | 0.939 |
| Random Forest | 0.858 | 0.802 | 0.819 | 0.810 | 0.936 |
| CatBoost | 0.856 | 0.841 | 0.756 | 0.796 | 0.932 |
| Bagging | 0.846 | 0.785 | 0.807 | 0.796 | 0.930 |
| XGBoost | 0.852 | 0.831 | 0.756 | 0.792 | 0.930 |

Esta fase sirvio como benchmark inicial. El mejor finalista sin Optuna fue LightGBM:

- F1-score: 0.813
- ROC-AUC: 0.939
- Artefacto: `models/finalists_full/best_model.joblib`

Estos resultados quedaron superados por los modelos finales optimizados con Optuna y por la separacion operativa entre pre-asignacion y post-asignacion descrita al inicio de esta seccion.

Los graficos del benchmark de los 15 modelos se guardan en `outputs/benchmark_15/`, incluyendo matrices de confusion, curvas ROC individuales, curva ROC comparativa e importancia de variables.

## 8. Conclusiones

El proyecto queda finalmente con dos modelos LightGBM, cada uno pensado para un momento distinto del proceso de reserva:

- Modelo pre-asignacion: se usa antes de conocer la habitacion asignada. Es el modelo mas honesto para prediccion temprana y alcanza F1 `0.8473` y ROC-AUC `0.9559`.
- Modelo post-asignacion: se usa cuando ya existe `assigned_room_type` y puede calcularse `room_changed`. Es ligeramente mas preciso, con F1 `0.8521` y ROC-AUC `0.9586`.

La decision de separar ambos escenarios evita mezclar variables disponibles en momentos diferentes del negocio. `reservation_status` y `reservation_status_date` quedan excluidas en todos los casos porque representan data leak directo. `room_changed`, en cambio, no se usa en el modelo pre-asignacion porque depende de una decision posterior, pero si se permite en el modelo post-asignacion.

El benchmark inicial de 15 modelos fue util para seleccionar la familia de modelos. LightGBM ofrecio el mejor equilibrio entre precision, recall, F1 y ROC-AUC. Random Forest quedo como alternativa robusta, pero no supero a LightGBM tras el ajuste con Optuna.

## 9. Limitaciones y mejoras

- El modelo post-asignacion es mas preciso, pero no debe usarse antes de asignar habitacion.
- El modelo pre-asignacion es el recomendable para accion comercial temprana, aunque sacrifica una pequena parte de precision frente al escenario post-asignacion.
- Extender Optuna a XGBoost y CatBoost para comprobar si pueden superar a LightGBM con busquedas largas.
- Registrar el benchmark en MLflow para comparar ejecuciones y parametros.
- Ampliar la API FastAPI con autenticacion, versionado formal de modelos y validacion estricta del contrato de entrada.
- Preparar una version Streamlit para explicar resultados en la defensa.
