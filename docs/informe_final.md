# Informe final - Prediccion de cancelaciones hoteleras

## 1. Participantes

- Victor "victorh1397@gmail.com"
- Alejandro Aguado "alejandro.aguado@gmail.com"

## 2. Analisis inicial

Principales datos:

- El valor objetivo tiene desbalance moderado: aproximadamente 63% reservas no canceladas y 37% canceladas.
- `lead_time` muestra una relacion importante con la cancelacion: las reservas hechas con mas antelacion tienden a cancelarse mas.
- `reservation_status` y `reservation_status_date` se eliminaron por ser un "data leak" , ya que representan informacion posterior al estado final de la reserva.
- `agent` y `company` se eliminaron por ser identificadores administrativos con muchos nulos.
- Se detecto un valor negativo en `adr`, que se descarta por incoherencia de negocio.
- Los nulos en `children` y `country` son porcentualmente bajos, por lo que se eliminan.

## 3. Decisiones sobre dataset

Se mantienen dos versiones de modelado para poder compararlas sin pisarse.

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

## 4. Diseno del sistema

El proyecto se organiza como un pipeline ejecutable:

```text
src/data.py      -> carga, limpieza, feature engineering y split X/y
src/models.py    -> definicion de modelos y preprocesador comun
src/evaluate.py  -> metricas, graficos y artefactos
src/train.py     -> CLI que orquesta el flujo completo
src/api.py       -> API FastAPI para inferencia con modelos guardados
tests/           -> pruebas de limpieza, features, entrenamiento, API y backend grafico
```

El preprocesador usa `ColumnTransformer` con:

- `OneHotEncoder(handle_unknown='ignore', drop='first')` para categoricas.
- `StandardScaler` para numericas.

La seleccion del modelo se realiza segun F1-score, manteniendo tambien accuracy, precision, recall y ROC-AUC para comparacion.

El CLI permite seleccionar la version con `--feature-set victor` o `--feature-set alejandro`. Tambien permite guardar el CSV usado con `--processed-data-path`.

Como mejora tecnica adicional, se incluye una API FastAPI para servir inferencia:

- `GET /health`: validacion basica de servicio.
- `GET /models`: listado de modelos `.joblib` disponibles.
- `POST /predict/{model_name}`: prediccion con el modelo seleccionado y el feature set indicado.

Se puede levantar con:

```powershell
$env:HOTEL_MODELS_DIR = "models/finalists_full"
uvicorn src.api:app --reload
```

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

Ademas, se incluye balanceo con `class_weight` en los modelos que lo soportan y opcion `--tune` para usar `GridSearchCV` en modelos scikit-learn.

## 6. Metrica principal

La metrica principal es F1-score. La razon es que el dataset no esta perfectamente balanceado y el coste de negocio no se concentra solo en una clase:

- Un falso negativo implica no detectar una reserva que se cancelara.
- Un falso positivo puede activar acciones comerciales innecesarias.

F1 equilibra precision y recall, y por tanto es mas informativa que accuracy para tomar esta decision.

## 7. Resultados

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

El modelo elegido por F1 fue LightGBM:

- F1-score: 0.813
- ROC-AUC: 0.939
- Artefacto: `models/finalists_full/best_model.joblib`

Los graficos del benchmark de los 15 modelos se guardan en `outputs/benchmark_15/`, incluyendo matrices de confusion, curvas ROC individuales, curva ROC comparativa e importancia de variables.

## 8. Conclusiones

LightGBM ofrece el mejor equilibrio entre precision y recall tras ampliar el benchmark a 15 modelos. En la validacion sobre todo el dataset alcanza F1 `0.813` y ROC-AUC `0.939`, superando ligeramente a Random Forest, que queda como segunda opcion robusta con F1 `0.810`.

CatBoost, Bagging y XGBoost tambien muestran resultados competitivos, aunque con menor recall que LightGBM.

La comparacion de datasets deja dos opciones defendibles: `victor`, que mantiene  mejor ROC-AUC, y `alejandro`, que tiene F1 ligeramente superior y menor riesgo de "data leak" por no usar `assigned_room_type`.

## 9. Limitaciones y mejoras

- Optimizar hiperparametros de LightGBM, XGBoost y CatBoost con validacion cruzada.
- Registrar el benchmark en MLflow para comparar ejecuciones y parametros.
- Ampliar la API FastAPI con autenticacion, versionado formal de modelos y validacion estricta del contrato de entrada.
- Preparar una version Streamlit para explicar resultados en la defensa.
