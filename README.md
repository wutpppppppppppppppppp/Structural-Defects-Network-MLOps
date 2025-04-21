# Structural-Defects-Network-MLOps

## 📦 Setup Instructions

### 1. (Optional) Create a Virtual Environment
We recommend using a virtual environment to manage dependencies.

**For Windows:**
```bash
python -m venv env
.\env\Scripts\activate
```
**For macOS/Linux:**
```bash
python -m venv env
source env/bin/activate
```

### 2. Install Required Packages
Install all required Python packages using the requirements.txt file:
```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing
Run the preprocessing script before training or evaluation:
```bash
python preprocess.py
```


### 4. Data Artifacts
After running the preprocessing script:
The train, test, and validation datasets will be saved inside the artifact_folder/ directory.
The artifact_folder/ is excluded from version control and will not be uploaded to Git (as specified in .gitignore).

```kotlin
Structural-Defects-Network-MLOps/
├── Dataset             # Raw Data
├── src/                
│   └── preprocess.py/ 
├── requirements.txt
├── .gitignore
├── README.md
└── artifact_folder/     # Contains train/test/val data (ignored by Git)
```