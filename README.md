# 😊 Facial Expression Recognition using CNN | FER2013 Dataset

This project aims to classify human facial expressions using Convolutional Neural Networks (CNNs) trained on the FER2013 dataset. It was originally developed in a Kaggle Notebook and structured into a modular Python project for easier maintenance, scalability, and deployment.

---

## 💡 About the Project

Facial expressions are a core component of non-verbal communication. With the increasing integration of AI into human-centric applications — from security systems to emotion-aware virtual assistants — recognizing emotions from faces has become a valuable deep learning task.

This project uses deep learning to classify grayscale 48x48 pixel facial images into seven categories:

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😀 Happy
- 😢 Sad
- 😲 Surprise
- 😐 Neutral

---

## 🗂 Folder Structure

```
fer2013-cnn/
├── data_preprocessing/       # Data cleaning, reshaping, and splitting
│   └── prepare_data.py
├── training/                 # CNN model architecture and training loop
│   └── train_model.py
├── validation/               # Evaluation logic and result visualization
│   └── evaluate_model.py
├── output/                   # Saved models, plots, and result files
├── fer2013-cnn.ipynb         # Complete original Kaggle notebook
├── requirements.txt          # All required libraries
└── README.md                 # This file
```

---

## 📦 Dataset

- 📍 **Name**: FER2013 (Facial Expression Recognition 2013)
- 📥 [Download from Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- 📄 Format: CSV with columns for emotion, pixel data (as string), and usage (train/test/val)

---

## ⚙️ How to Run the Project

> Ensure Python 3.x and pip are installed on your machine.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess the Dataset

Convert the CSV pixel strings into 48x48 grayscale images and split them:

```bash
python data_preprocessing/prepare_data.py
```

### 3. Train the CNN Model

```bash
python training/train_model.py
```

### 4. Evaluate Model Performance

```bash
python validation/evaluate_model.py
```

---

## 📊 Sample Output

- ✅ Accuracy and loss plots
- 🧠 Trained `.h5` model
- 🔍 Confusion matrix and classification report

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Pandas, NumPy, Matplotlib
- Scikit-learn

---

## 🤝 Contributions

Pull requests are welcome. Feel free to fork the repo and propose changes.

---

## 📄 License

MIT License – do anything you want with attribution.

---

## 🙋‍♀️ Author

Developed by **Aahana Kaur**  
Feel free to connect on [LinkedIn](https://linkedin.com) or reach out for collaborations.
