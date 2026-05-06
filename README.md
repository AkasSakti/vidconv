# Deteksi dan Counting Bungkus Produk Minuman di Colab

Aplikasi ini disiapkan untuk struktur Google Colab/Drive:

```text
/content/drive/My Drive/Colab Notebooks/vidconv
```

Folder lokal ini hanya simulasi struktur Drive agar kode mudah dibuat dan diuji. Dataset awal hanya 1 gambar per kelas, sehingga proyek menyediakan generator data sintetis agresif dalam format YOLO.

## Struktur

- `vidconv_colab.ipynb`: notebook utama untuk Google Colab.
- `app.py`: UI Streamlit, deteksi YOLOv8, tracking, counting, unknown log, dan grafik live.
- `src/dataset_generator.py`: augmentasi dan synthetic data generation.
- `train_yolo.py`: training YOLOv8n custom dataset.
- `models/best.pt`: model hasil training untuk deployment Streamlit Cloud via GitHub.
- `dataset/`: gambar awal per kelas.
- `generated_dataset/`: output dataset YOLO hasil augmentasi.

## Setup Colab

Buka dan jalankan notebook:

```text
vidconv_colab.ipynb
```

Atau jalankan cell manual berikut:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd "/content/drive/My Drive/Colab Notebooks/vidconv"
!pip install -q "cryptography>=42,<44" "pyOpenSSL>=24.0.0,<=24.2.1"
!pip install -q -r requirements.txt --upgrade-strategy only-if-needed
```

## Generate Dataset YOLO di Colab

```bash
python -m src.dataset_generator --samples-per-class 200
```

Output:

```text
generated_dataset/
  images/train
  images/val
  labels/train
  labels/val
  dataset.yaml
```

## Training YOLOv8n di Colab

```bash
python train_yolo.py --data generated_dataset/dataset.yaml --epochs 80 --batch 8 --model yolov8n.pt
```

Model terbaik disimpan di:

```text
runs/beverage_yolov8n/weights/best.pt
```

## Serving Streamlit via GitHub

Setelah training di Colab selesai, salin model terbaik ke path deployment:

```bash
mkdir -p models
cp runs/beverage_yolov8n/weights/best.pt models/best.pt
```

Push project ke GitHub, lalu deploy di Streamlit Cloud dengan main file:

```text
app.py
```

Jika Streamlit Cloud menampilkan pilihan Python manual, pilih Python `3.12`. Untuk mengubah versi Python pada app yang sudah pernah dibuat, hapus app dari dashboard lalu deploy ulang dengan pilihan Python yang benar.

Fitur aplikasi:

- Mode `Browser camera - GitHub/Streamlit Cloud` berbasis WebRTC.
- Mode `OpenCV camera - local runtime` untuk testing lokal.
- Bounding box dan label class.
- Centroid tracking sederhana.
- Line crossing horizontal untuk counting.
- Pencegahan double counting berbasis ID tracking.
- Filter unknown untuk confidence rendah, area bbox terlalu kecil, atau aspect ratio tidak wajar.
- Jumlah per produk, unknown, total objek, error rate, log timestamp/confidence/reason, dan grafik live.

## Catatan Kamera

Untuk Streamlit Cloud via GitHub, jangan pakai `cv2.VideoCapture(0)` sebagai mode utama. Aplikasi sudah menyediakan mode browser camera berbasis WebRTC agar kamera dibuka dari browser pengguna.

`cv2.VideoCapture(0)` membaca kamera pada mesin yang menjalankan Python. Di hosted runtime Colab, mesin Python berada di server Google, bukan laptop, sehingga webcam laptop tidak langsung terbaca oleh OpenCV.

Pilihan yang benar:

- Untuk GitHub/Streamlit Cloud, gunakan mode `Browser camera - GitHub/Streamlit Cloud`.
- Untuk lokal atau Colab local runtime, mode `OpenCV camera - local runtime` bisa dipakai.

Perintah Colab lengkap:

```python
%cd "/content/drive/My Drive/Colab Notebooks/vidconv"
!pip install -q "cryptography>=42,<44" "pyOpenSSL>=24.0.0,<=24.2.1"
!pip install -q -r requirements.txt --upgrade-strategy only-if-needed
!python -m src.dataset_generator --samples-per-class 200
!python train_yolo.py --data generated_dataset/dataset.yaml --epochs 80 --batch 8
!mkdir -p models
!cp runs/beverage_yolov8n/weights/best.pt models/best.pt
```
