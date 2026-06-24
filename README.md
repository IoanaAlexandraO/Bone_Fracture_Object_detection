# Detecție Fracturi Osoase — YOLOv8s

Sistem automat de detecție a fracturilor osoase din radiografii, bazat pe YOLOv8s.

## Clase detectate
`elbow positive` · `fingers positive` · `forearm fracture` · `humerus fracture` · `humerus` · `shoulder fracture` · `wrist positive`

## Rezultate
| Metrică | Valoare |
|---|---|
| mAP@0.5 | 0.266 |
| Precision | 0.249 |
| Recall | 0.327 |

## Rulare

```bash
pip install ultralytics streamlit
streamlit run app.py
```

## Dataset
[Bone Fracture Detection — Kaggle](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project)
