# Week 3 YOLOv8 experiments

All models are YOLOv8n trained on an 8 GB Apple M1 and scored by `week03/detmetrics.py` on the test split (COCO style AP).

| Experiment | Split | Inference | mAP50 | mAP50-95 | Ball AP50 | Goalkeeper AP50 | Player AP50 | Referee AP50 | ms / frame | Train time |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `notebook-split` | original | whole frame 640 px | 0.609 | 0.395 | 0.000 | 0.844 | 0.965 | 0.628 | 77 | 39 min |
| `baseline` | clip | whole frame 640 px | 0.588 | 0.382 | 0.000 | 0.861 | 0.962 | 0.530 | 60 | 23 min |
| `offline-aug` | clip | whole frame 640 px | 0.580 | 0.347 | 0.000 | 0.776 | 0.944 | 0.599 | 56 | 21 min |
| `hires-inference` | clip | whole frame 1280 px | 0.419 | 0.268 | 0.000 | 0.142 | 0.943 | 0.590 | 130 | 0 min |
| `tiles` | clip | whole frame 640 px | 0.380 | 0.215 | 0.100 | 0.216 | 0.896 | 0.308 | 47 | 24 min |
| `tiles` | clip | sliced 640 px | 0.820 | 0.567 | 0.569 | 0.945 | 0.965 | 0.799 | 248 | 24 min |

Experiments:

* `notebook-split`: Notebook recipe on Roboflow's random split (clips leak into test).
* `baseline`: Notebook recipe on the leakage free clip split.
* `offline-aug`: Plus Roboflow style offline augmentation (2 extra copies), equal step budget.
* `hires-inference`: Baseline weights, inference at 1280 px.
* `tiles`: Train on 960x540 tiles at 640 px, equal step budget, sliced inference.
