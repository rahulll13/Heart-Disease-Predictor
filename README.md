# Heart Shield 🛡️ 

**Advanced AI-Powered Early Heart Disease Risk Predictor**

---

## 📌 Overview

**Heart Shield** is a secure, interactive web application designed to assess the probability of heart disease risk based on patient medical data. Built with a focus on clean design and data privacy, the application utilizes a Machine Learning model to provide instant, real-time risk assessments while ensuring all processing happens locally during the user session.

> ⚠️ **Medical Disclaimer:**  
> This project is intended **only for educational and informational purposes**. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional regarding any medical condition or health concern. If you believe you are experiencing a medical emergency, contact your local emergency services immediately.

---

# 🚀 Key Features

- 🧠 **AI-Powered Prediction**
  - Uses a trained **Random Forest Classifier** to analyze **13 important clinical parameters** for heart disease prediction.

- ⚡ **Instant Risk Assessment**
  - Generates real-time predictions with clear categorization into:
    - ✅ Low Risk
    - ⚠️ High Risk

- 📊 **Probability Score**
  - Displays prediction confidence along with personalized health recommendations.

- 🛡️ **Privacy First**
  - No user medical information is stored.
  - All prediction processing occurs during the active session.

- 🎨 **Modern User Interface**
  - Clean, responsive, and light-themed interface built with Streamlit.
  - Easy-to-use sidebar for patient data entry.

- 📱 **Cross Platform**
  - Accessible through any modern web browser.

---

# 🛠 Tech Stack

| Component | Technology |
|------------|------------|
| **Frontend** | Streamlit, HTML, CSS |
| **Backend** | Python 3 |
| **Machine Learning** | Scikit-learn (Random Forest Classifier) |
| **Data Processing** | Pandas, NumPy |
| **Model Serialization** | Pickle |

---

# 📥 Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/rahulll13/Heart-diseaseprediction.git
cd Heart-diseaseprediction
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the Application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser.

---

# 📂 Project Structure

```text
Heart-diseaseprediction/
│
├── .streamlit/
│   └── config.toml              # Streamlit configuration
│
├── app.py                       # Main application
├── heart_disease_model.pkl      # Trained Random Forest model
├── scaler.pkl                   # StandardScaler object
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
│
└── assets/                      # Optional screenshots/images
    ├── home.png
    ├── prediction.png
    └── about.png
```

---

# 📊 How It Works

### Step 1 — User Input

The user enters **13 clinical parameters**, including:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression (Oldpeak)
- Slope of ST Segment
- Number of Major Vessels
- Thalassemia

---

### Step 2 — Data Preprocessing

The input values are transformed using a pre-trained **StandardScaler** to ensure consistency with the model's training data.

---

### Step 3 — AI Prediction

The processed input is passed to the trained **Random Forest Classifier**, which predicts:

- **0 → Low Risk**
- **1 → High Risk**

The model also computes the prediction probability.

---

### Step 4 — Results

The application displays:

- Prediction result
- Risk level
- Prediction confidence
- Helpful health recommendations
- Medical disclaimer

---

# 📈 Machine Learning Model

**Algorithm Used**

- Random Forest Classifier

### Why Random Forest?

- High prediction accuracy
- Handles non-linear relationships
- Resistant to overfitting
- Works well with medical datasets
- Provides probability estimates

---

# 🔒 Privacy & Security

Heart Shield follows a **Privacy-First** approach.

✅ No database storage

✅ No user authentication required

✅ No medical records saved

✅ Local prediction during active session

---

# 📷 Screenshots

## 🏠 Home & Patient Data Entry

![Home Screen](https://raw.githubusercontent.com/rahulll13/Heart-Disease-Predictor/main/assets/Home.png.png)

## 📊 Prediction Results

![Prediction Results](https://github.com/rahulll13/Heart-Disease-Predictor/blob/main/assets/prediction.png.png)

---

## ℹ️ About the Application

![About Application ](https://github.com/rahulll13/Heart-Disease-Predictor/blob/main/assets/about.png.png)

---

# 🎯 Future Improvements

- Explainable AI (SHAP/LIME)
- PDF Health Report Generation
- User Authentication
- Electronic Health Record (EHR) Integration
- Doctor Dashboard
- Model Comparison
- Cloud Deployment
- Mobile-Friendly Interface
- Multi-language Support

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve this project:

1. Fork the repository

2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Added new feature"
```

4. Push your branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📜 License

This project is licensed under the **MIT License**.

Feel free to use, modify, and distribute this project with proper attribution.

---

# 👨‍💻 Developer

## Rahul Kumar Sinha

**B.Tech in Information Technology**  

Passionate about building AI-driven solutions that solve real-world problems through Machine Learning, Full Stack Development, and Intelligent Automation.

### Experience

- AI Internship — Infosys Springboard 6.0
- Cyber Security & Ethical Hacking Internship — C-DAC, Noida

### Technical Skills

- Python
- Flask
- React
- MySQL
- TensorFlow
- Scikit-learn
- Pandas
- NumPy

---

## 🌐 Connect With Me

**GitHub**

https://github.com/rahulll13

**Email**

sinha.rahul2318@gmail.com
---

## ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub.

Your support motivates further development and improvements.

---

> **"Advancing Healthcare Accessibility Through Artificial Intelligence."**

**Made with ❤️ by Rahul Kumar Sinha**
