# Structural-Defects-Network-MLOps

## 📦 Setup Instructions

### 1. (Optional) Create a Virtual Environment
We recommend using a virtual environment to manage dependencies.

**For Windows:**
```bash
python -m venv env
.\env\Scripts\activate

2. Install Required Packages
Install all required Python packages using the requirements.txt file:

pip install -r requirements.txt

3. Run Preprocessing
Run the preprocessing script before training or evaluation:

python preprocess.py

4. Data Artifacts
After running the preprocessing script:
The train, test, and validation datasets will be saved inside the artifact_folder/ directory.
The artifact_folder/ is excluded from version control and will not be uploaded to Git (as specified in .gitignore).