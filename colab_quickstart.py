"""
Quickstart cells for Google Colab.

Copy each block into a Colab notebook, or run the equivalent commands from a
Colab cell after the project has been placed in:
/content/drive/My Drive/Colab Notebooks/vidconv
"""


MOUNT_DRIVE = """
from google.colab import drive
drive.mount('/content/drive')
%cd "/content/drive/My Drive/Colab Notebooks/vidconv"
"""


INSTALL = """
!pip install -r requirements.txt
"""


GENERATE_DATASET = """
!python -m src.dataset_generator --samples-per-class 200
"""


TRAIN = """
!python train_yolo.py --data generated_dataset/dataset.yaml --epochs 80 --batch 8 --model yolov8n.pt
"""


PREPARE_MODEL_FOR_GITHUB = """
!mkdir -p models
!cp runs/beverage_yolov8n/weights/best.pt models/best.pt
!ls -lh models/best.pt
"""


GITHUB_STREAMLIT_NOTE = """
Serving target:
Push app.py, src/, requirements.txt, .streamlit/config.toml, and models/best.pt
to GitHub, then deploy the repo from Streamlit Cloud with app.py as the main
file. Use the app mode "Browser camera - GitHub/Streamlit Cloud".
"""
