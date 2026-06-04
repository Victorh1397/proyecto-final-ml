# Decisiones Sobre Datasets

Este documento recoge las decisiones tomadas durante la preparacion de los datasets del proyecto de prediccion de cancelaciones hoteleras.

## Versiones mantenidas

- `dataset_practica_final_modelado.csv`: version de referencia asociada al trabajo original de Victor.
- `dataset_practica_final_modelado_alejandro.csv`: version alternativa de Alejandro, ya entregada y congelada.
- `dataset_practica_final_modelado_binary.csv`: version experimental activa para nuevas pruebas.

Victor y Alejandro se mantienen sin modificaciones adicionales para no alterar entregables ya revisados. Las pruebas nuevas se concentran en `binary`.

## Fugas de informacion eliminadas

Se excluyen columnas que no estarian disponibles o que revelan el resultado de la reserva:

- `reservation_status`
- `reservation_status_date`

Tambien se excluyen identificadores administrativos:

- `agent`
- `company`

## Decision sobre `assigned_room_type`

En la version Alejandro y en la version binary se elimina `assigned_room_type`.

Motivo: si la prediccion se hace en el momento de crear la reserva, la habitacion asignada puede no existir todavia o puede depender de decisiones operativas posteriores. Preferimos usar `reserved_room_type`, que si forma parte de la reserva inicial.

## Decision sobre variables derivadas sospechosas

Se descartan variables derivadas como:

- `total_price`
- `room_changed`
- `has_agent`
- `has_company`
- `previous_cancel_ratio`

Motivo: algunas no tienen suficiente sentido de negocio en el momento de prediccion y otras pueden introducir informacion administrativa o derivada de decisiones posteriores. La mejora de metricas no compensa el riesgo de data leak.

## Decision sobre outliers

Victor y Alejandro no se modifican.

En `binary` se aplican reglas conservadoras:

- `adr <= 1000`
- `total_guests > 0`
- `total_guests <= 10`

Motivo: se detectaron valores extremos , como `adr = 5400`, reservas con cero huespedes y reservas con decenas de adultos. El filtrado afecta a un volumen muy pequeno del dataset y se limita a la variante experimental.

## Decision sobre binary

La primera version binary sustituyo demasiadas variables numericas por indicadores binarios. El resultado bajo mucho, por lo que se descarto ese enfoque.

La version actual de binary conserva las variables originales y agrega indicadores simples de negocio:

- `has_previous_cancellations`
- `has_previous_bookings`
- `has_waiting_days`
- `is_weekend_stay`
- `is_long_stay`
- `is_high_adr`
- `is_last_minute_booking`
- `is_early_booking`
- `is_non_refundable`
- `is_portugal`
- `is_online_ta`
- `is_group_segment`

Motivo: LightGBM aprovecha bien variables numericas continuas como `lead_time` y `adr`. Los binarios se usan como señales adicionales, no como sustitutos de la granularidad.

## Decision sobre cardinalidad

En binary se reduce la cardinalidad de variables categoricas:

- Se elimina `country` original.
- Se conserva `country_grouped`, limitado a los 15 paises mas frecuentes.
- Se conserva `is_portugal`, porque Portugal tiene mucha señal en este dataset.
- `reserved_room_type` conserva `A`, `D`, `E`, `F` y `G`; el resto se agrupa como `Other`.

Motivo: reducir columnas generadas por one-hot encoding y evitar categorias raras con poca frecuencia, sin perder las señales de negocio principales.

## Decision sobre codificacion de categoricas

Se probaron varias estrategias de codificacion para LightGBM sobre binary:

- One-hot encoding actual.
- Ordinal encoding.
- Categoricas nativas de LightGBM.
- Target mean encoding ajustado solo sobre train.
- Basicamente todo lo que nos recomendo Claude

Resultado en muestra de 10.000 filas:

| Estrategia | F1 | ROC-AUC |
| --- | ---: | ---: |
| Target mean encoding | 0.7875 | 0.9221 |
| Ordinal encoding | 0.7837 | 0.9188 |
| Categoricas nativas LightGBM | 0.7787 | 0.9159 |
| One-hot actual | 0.7767 | 0.9166 |

Resultado en dataset completo binary:

| Estrategia | F1 | ROC-AUC |
| --- | ---: | ---: |
| Categoricas nativas LightGBM | 0.8073 | 0.9345 |
| One-hot actual | 0.8064 | 0.9334 |
| Ordinal encoding | 0.8061 | 0.9336 |
| Target mean encoding | 0.8041 | 0.9338 |

Conclusion: las categoricas nativas de LightGBM mejoran ligeramente en el dataset completo, pero la mejora es pequeña. Target encoding mejora en muestra pequeña, pero no en full data.

## Modelo post-asignacion

Se probo un feature set llamado `post_asignacion` con columnas disponibles o derivables despues de la asignacion de habitacion:

- `total_guests`
- `total_nights`
- `adr_per_person`
- `previous_cancel_ratio`
- `room_changed`
- `has_agent`
- `has_company`
- `arrival_season`

En esta variante solo se eliminan directamente `reservation_status`, `reservation_status_date` y `arrival_date_year`. Los identificadores `agent`, `company` y `assigned_room_type` se usan para crear flags y despues se eliminan como columnas brutas.

Resultados con LightGBM:

| Variante | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `post_asignacion` base | 0.8579 | 0.7812 | 0.8574 | 0.8175 | 0.9418 |
| `post_asignacion` con parametros Optuna de Victor | 0.8888 | 0.8418 | 0.8625 | 0.8521 | 0.9586 |
| `post_asignacion` sin `room_changed` | 0.8831 | 0.8345 | 0.8547 | 0.8445 | 0.9548 |
| `post_asignacion` sin `has_agent`/`has_company` | 0.8882 | 0.8412 | 0.8617 | 0.8513 | 0.9584 |
| `post_asignacion` sin `room_changed` ni flags admin | 0.8824 | 0.8347 | 0.8520 | 0.8433 | 0.9544 |

Conclusion: el set post-asignacion completo supera el mejor resultado anterior (`F1` 0.8521 frente a 0.8473), pero la mejora depende principalmente de conservar `room_changed`. Si `room_changed` se elimina, la variante cae por debajo del modelo anterior. Por tanto, solo seria utilizable si el caso de uso permite predecir despues de que exista habitacion asignada; si la prediccion debe hacerse en el momento inicial de la reserva, `room_changed` debe considerarse un data leak.

## Decision sobre dos momentos de prediccion

Se plantea mantener dos modelos segun el momento operativo:

- Modelo antes de asignar habitacion: no debe usar `assigned_room_type` ni `room_changed`. Es el escenario mas real para prediccion temprana.
- Modelo despues de asignar habitacion: puede usar `room_changed`, porque en ese momento ya se conoce si la habitacion asignada coincide con la reservada y da mejores resultados que el anterior.

Motivo: no todos los campos tienen la misma disponibilidad temporal. Separar los modelos evita mezclar un modelo preventivo con otro operativo posterior. La API ya permite seleccionar el `feature_set`, por lo que el flujo puede elegir el modelo correcto segun el momento de inferencia.

Se programa una busqueda Optuna de 24 horas sobre el modelo posterior (`feature_set=post_asignacion`) para comprobar si el resultado mejora al optimizar hiperparametros especificamente para estas columnas.

Resultado de la busqueda Optuna de 24 horas:

| Variante | Trials | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `post_asignacion` Optuna 8h | 661 | 0.8882 | 0.8410 | 0.8620 | 0.8514 | 0.9584 |

Mejor trial de validacion cruzada:

- Trial: `517`
- F1 CV: `0.8458`
- `n_estimators`: `1900`
- `learning_rate`: `0.014839095842342522`
- `num_leaves`: `240`
- `max_depth`: `16`
- `min_child_samples`: `10`
- `subsample`: `0.9033910785111278`
- `colsample_bytree`: `0.7960451523278428`
- `class_weight`: `balanced`

Conclusion posterior: Optuna 24h mejora al modelo Victor Optuna anterior (`F1` 0.8514 frente a 0.8473), pero no supera la prueba manual que reutilizaba los parametros Optuna de Victor sobre `post_asignacion` (`F1` 0.8521). La diferencia es pequeña, por lo que ambos son candidatos cercanos para el modelo post-asignacion.

## Decision sobre balanceo de clases

La clase positiva (`is_canceled = 1`) representa aproximadamente el 37% del dataset, por lo que existe desbalance, pero no es extremo.

Se probaron varias estrategias de balanceo con LightGBM y categoricas nativas sobre binary:

| Estrategia | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| `class_weight="balanced"` | 0.7818 | 0.8344 | 0.8073 | 0.9345 |
| `scale_pos_weight=neg/pos` | 0.7786 | 0.8352 | 0.8059 | 0.9345 |
| `is_unbalance=True` | 0.7786 | 0.8352 | 0.8059 | 0.9345 |
| Submuestreo aleatorio | 0.7772 | 0.8342 | 0.8047 | 0.9334 |
| Sobremuestreo aleatorio | 0.7793 | 0.8308 | 0.8043 | 0.9331 |
| `scale_pos_weight=sqrt(neg/pos)` | 0.8109 | 0.7973 | 0.8040 | 0.9340 |
| Sin balanceo | 0.8403 | 0.7578 | 0.7969 | 0.9342 |

Conclusion: se mantiene `class_weight="balanced"` como opcion preferente para maximizar F1. Sin balanceo mejora precision y accuracy, pero pierde demasiado recall en cancelaciones. `scale_pos_weight` e `is_unbalance` suben ligeramente el recall, pero no mejoran F1.

## Decision sobre control de overfitting en LightGBM

Se probo una parrilla local sobre binary con categoricas nativas de LightGBM, manteniendo `objective="binary"` y `class_weight="balanced"`.

El mejor resultado fue:

| Parametro | Valor |
| --- | ---: |
| `n_estimators` | 220 |
| `learning_rate` | 0.06 |
| `num_leaves` | 127 |
| `max_depth` | -1 |
| `min_child_samples` | 20 |
| `feature_fraction` | 0.75 |
| `bagging_fraction` | 0.90 |
| `bagging_freq` | 1 |
| `reg_alpha` | 0.0 |
| `reg_lambda` | 0.0 |

Metricas del mejor experimento:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0.8678 | 0.8070 | 0.8468 | 0.8264 | 0.9449 |

Conclusion: controlar la complejidad y el muestreo por arbol tiene mucho mas impacto que convertir variables a binarias. La mejor combinacion encontrada aumenta bastante el F1 frente al LightGBM nativo base (`0.8073` a `0.8264`).

Validacion cruzada posterior con 5 folds:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM nativo base | 0.8554 | 0.7830 | 0.8454 | 0.8130 | 0.9378 |
| LightGBM nativo ajustado | 0.8699 | 0.8059 | 0.8564 | 0.8303 | 0.9480 |
| Mejora | +0.0145 | +0.0229 | +0.0110 | +0.0174 | +0.0102 |

Conclusion final: esta mejora en binary fue valida durante la fase experimental, pero no se selecciona como modelo final porque queda por debajo de los dos modelos definitivos. El proyecto se cierra con dos modelos segun el momento de prediccion:

- Modelo pre-asignacion: `feature_set=victor`, sin `room_changed`, F1 `0.8473`.
- Modelo post-asignacion: `feature_set=post_asignacion`, usando `room_changed` cuando la habitacion ya esta asignada, F1 `0.8521`.

Por tanto, binary queda documentado como experimento descartado.

## Criterio general

La prioridad es mantener un dataset creible:

- Variables disponibles en el momento de prediccion.
- Transformaciones simples de explicar.
- Sin fugas directas ni columnas administrativas dudosas.
- Cambios experimentales aislados en binary.
