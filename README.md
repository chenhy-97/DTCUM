# DTCUM
 Deep-Learning Time-series Carotid Ultrasound for MACE Prediction Model (DTCUM), a system for MACE, MACE Subgroups, MACE Probability Prediction, and Weight Analysis.

![微信图片_20240328104701](https://github.com/chenhy-97/DTCUM/assets/74726912/42b29e48-aff2-49c3-88f1-4f403d6c4c1b)

### Data Preprocessing
Move all patient folders for each year to the '/img' directory, and rename the folders for left intima, left plaque, right intima, and right plaque respectively as '11.png', '12.png', '21.png', and '22.png'.

```
img/
│
├── 1/
│   ├── 11.png
│   ├── 12.png
│   ├── 21.png
│   └── 22.png
...
│
└── 5/
    ├── 11.png
    ├── 12.png
    ├── 21.png
    └── 22.png
```

### Install Dependencies

```
pip install torch pandas pillow
```

### Predicting MACE Event

```
python vaild.py
```

### Pretrained Weight (Preparing)

| Model       | Link   | 
| :--------  | :-----  |
| Feature Extraction |  |
| Transformer |  |
