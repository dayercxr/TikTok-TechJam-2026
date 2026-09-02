# VeriFrame

A full-stack prototype for classifying an uploaded image as AI-generated or likely real. 
This repository contains both the UI layer & the model handling the detection and classification

### Tech Stack
* React + Vite
* FastAPI
* PyTorch


### Project Structure
```bash
.
├── backend
│   ├── requirements.txt
│   └── src
│       ├── main.py
│       ├── model.py
│       └── train.py
├── frontend
│   ├── eslint.config.js
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── public
│   │   ├── favicon.svg
│   │   └── icons.svg
│   ├── README.md
│   ├── src
│   │   ├── App.css
│   │   ├── App.tsx
│   │   ├── assets
│   │   │   ├── hero.png
│   │   │   ├── react.svg
│   │   │   └── vite.svg
│   │   ├── index.css
│   │   └── main.tsx
│   ├── tsconfig.app.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
├── LICENSE
└── README.md
```

## Quick start
* Clone the repositor
```bash
git clone <>
```

### Setup Environment (Frontend)
* 

### Setup Environment (Backend)
* Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

* Install dependencies via CLI:
```bash
pip install -r requirements.txt
```

API runs at `http://localhost:8000`.




