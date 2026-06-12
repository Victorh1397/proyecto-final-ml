# Checklist de evaluacion

Estado revisado en local antes de subir cambios al repositorio remoto.

## Requisitos minimos

| Requisito | Estado | Evidencia |
| --- | --- | --- |
| Problema de clasificacion binaria | Cubierto | Variable objetivo `is_canceled` documentada en `README.md` y `docs/informe_final.md`. |
| Dataset real proporcionado | Cubierto | `data/raw/dataset_practica_final.csv`. |
| Justificacion del problema y datos | Cubierto | Secciones "Datos" y "Justificacion del problema". |
| Regresion logistica | Cubierto | `src/models.py`, `logistic_regression`. |
| Arbol de decision | Cubierto | `src/models.py`, `decision_tree`. |
| Random Forest | Cubierto | `src/models.py`, `random_forest`. |
| Gradient Boosting | Cubierto | `src/models.py`, `gradient_boosting` con XGBoost y modelos adicionales LightGBM/CatBoost. |
| Red neuronal multicapa Keras | Cubierto | `src/models.py`, `neural_network` usa TensorFlow/Keras cuando esta instalado. |
| Comparacion de modelos | Cubierto | `outputs/benchmark_15/metrics.csv` compara 15 modelos. |
| Metrica principal | Cubierto | F1-score justificado en README e informe. |
| Matriz de confusion | Cubierto | `outputs/benchmark_15/confusion_matrix_<modelo>.png`. |
| Curva ROC | Cubierto | `outputs/benchmark_15/roc_curve_<modelo>.png` y `roc_curves_comparison.png`. |
| Pipeline estructurado | Cubierto | `src/data.py`, `src/models.py`, `src/evaluate.py`, `src/train.py`, `src/tune_optuna.py`. |
| Carga, preprocesamiento, entrenamiento, evaluacion y seleccion | Cubierto | `python -m src.train ...` ejecuta el flujo completo. |

## Entregables obligatorios

| Entregable | Estado | Evidencia |
| --- | --- | --- |
| Repositorio GitHub | Pendiente de push | Los cambios estan en local. Hay que hacer commit y push al remoto antes de entregar. |
| Codigo bien estructurado | Cubierto | Carpetas `src/`, `tests/`, `notebooks/`, `data/`, `outputs/`, `models/`, API en `src/api.py` y tuning en `src/tune_optuna.py`. |
| README con autores | Cubierto | Victor y Alejandro figuran en `README.md`. |
| README con descripcion del problema y datos | Cubierto | Secciones "Datos" y objetivo del proyecto. |
| README con instrucciones de ejecucion | Cubierto | Seccion "Ejecucion". |
| README con resultados y conclusiones | Cubierto | Secciones "Resultados verificados" y "Conclusiones". |
| `.gitignore` adecuado | Cubierto | Ignora entornos, caches, logs y temporales. |
| `requirements.txt` | Cubierto | Incluye scikit-learn, TensorFlow, XGBoost, LightGBM, CatBoost, FastAPI, Uvicorn, Optuna y pytest. |
| Informe final PDF o Markdown | Cubierto | `docs/informe_final.md`. |
| Roles de la pareja | Cubierto en documentacion | Victor y Alejandro figuran en README e informe. |
| Analisis exploratorio de datos | Cubierto | `notebooks/exploracion/eda_inicial.ipynb` y resumen en informe. |
| Diseno del sistema | Cubierto | Informe y README describen pipeline y modulos. |
| Resultados y eleccion final | Cubierto | LightGBM elegido por F1. |
| Reflexion critica sobre limitaciones y mejoras | Cubierto | Seccion "Limitaciones y mejoras". |

## Puntos de calidad y riesgos revisados

| Punto | Estado | Evidencia |
| --- | --- | --- |
| Control de fuga de informacion | Cubierto | `reservation_status` y `reservation_status_date` eliminadas y documentadas como leak puro. |
| `reservation_status_date` casi 99.95% | Cubierto | Documentado en README e informe como rendimiento invalido por fuga. |
| Versiones de dataset comparables | Cubierto | `dataset_practica_final_modelado_victor.csv` y `dataset_practica_final_modelado_alejandro.csv`. |
| Guion para defensa | Local no versionado | `docs/guion_defensa.md` existe en local y esta excluido de Git por decision del equipo. |
| Pruebas automatizadas | Cubierto | `tests/`, verificado con `14 passed`. |

## Bonus tecnicos presentes

| Bonus | Estado | Evidencia |
| --- | --- | --- |
| Optimizacion de hiperparametros | Cubierto | `src/tune_optuna.py` optimiza LightGBM con Optuna y `docker-compose.optuna.yml` permite dejarlo 8 horas en Docker. |
| Balanceo de clases | Cubierto | `class_weight` en modelos compatibles. |
| Interpretabilidad | Cubierto | Graficos `feature_importance_<modelo>.png`. |
| Red neuronal Keras | Cubierto | TensorFlow/Keras instalado y probado. |
| API FastAPI | Cubierto | `src/api.py` con `GET /models` y `POST /predict/{model_name}`; probado en `tests/test_api.py`. |
| MLflow | No implementado | No es requisito minimo; queda como mejora futura. |

## Pendientes antes de entrega

1. Hacer commit de los cambios locales.
2. Subir los cambios al repositorio remoto de GitHub.
3. Si se quiere reflejar contribucion individual en historial, crear commits separados o documentar claramente la contribucion de Alejandro y Victor.
4. Convertir `docs/informe_final.md` a PDF si la plataforma exige PDF en lugar de Markdown.
