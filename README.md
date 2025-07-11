#  Gesture Vocalizer: A Vision-Based Assistive Communication System

**Project by:** Syeda Umaima Tamkeen

---

## 📝 Summary

The **Gesture Vocalizer** is a real-time, vision-based assistive communication system that converts **hand gestures** into **spoken language**. Designed for individuals with speech and hearing impairments, this system uses **computer vision** and **machine learning** to detect, classify, and vocalize static hand gestures using webcam input.

This project demonstrates how artificial intelligence and deep learning can bridge communication gaps for differently-abled communities, supporting Sustainable Development Goals (SDGs) for **inclusion** and **accessibility**.

---

## 🔍 Features

- 🖐️ Real-time hand gesture detection using **MediaPipe**
- 🧠 Gesture classification using **XGBoost** and **CNN**
- 📊 Preprocessed dataset with landmarks and image features
- 🔊 Text-to-speech conversion using **pyttsx3**
- 📈 Visualizations and performance analytics
- 🌐 Streamlit GUI for user interaction

---

## 📂 Project Structure

├── app.py # Main Streamlit interface
├── create_dataset.py # Data collection script
├── model.pkl / model_lp # Trained model files
├── visualizations/ # Charts, graphs, and heatmaps
├── xgboost_output/ # Output from XGBoost classifier
├── models/ # Model architecture/code
├── .idea/, .venv/ # VSCode & environment folders
├── *.py, *.csv, *.png, *.pickle # Supporting project files
├── requirements.txt # Python dependencies
└── README.md # This file


---

##  How It Works

1. **Hand Landmark Detection**  
   Uses **MediaPipe** to detect 21 key hand landmarks from the webcam feed or image.

2. **Feature Extraction**  
   Extracts angles/distances or image features to represent each gesture numerically.

3. **Model Prediction**  
   Trained models like **XGBoost**, **Random Forest**, or **CNN** classify gestures in real-time.

4. **Voice Output**  
   The recognized label is converted into audio using **pyttsx3**.

---

##  How to Run

###  Installation
```bash
git clone https://github.com/syeda-umaima-tamkeen/Gesture-Vocalizer-Assistive-Communication.git
cd Gesture-Vocalizer-Assistive-Communication
pip install -r requirements.txt

## ▶️ Run the App
streamlit run app.py

📊 Results
✅ High accuracy on 35 custom gesture classes

🎯 Real-time prediction speed with low latency

📉 Loss and accuracy graphs show strong model convergence

📊 Heatmaps and PCA plots demonstrate clear gesture separation

📌 Applications
🔇 Assistive technology for the speech and hearing impaired

🧠 Human-computer interaction

🤖 Smart homes or gesture-controlled devices

🏫 Educational tools for sign language learning

🛠️ Tech Stack
Python 3.9+

MediaPipe – hand landmark detection

XGBoost / CNN – gesture classification

Pyttsx3 – text-to-speech engine

Streamlit – web app deployment

Matplotlib / Seaborn – visualization

Pickle – model serialization

🏁 License
This project is licensed under the MIT License 

