# Cotton-plant-disease-prediction-using-CNN-and-Flask
It uses  a deep learning model (InceptionV3 based CNN) to  classify cotton leaf diseases. The application is deployed using Flask, where users can upload an image and get predictions instantly.
Done. I’ll give you a clean, professional, interviewer-proof README that makes your project look legit and technically solid — without needing to upload the .h5 file.

Use this exact README.md in your GitHub repo.

⸻

🌿 Cotton Disease Prediction — README.md (Copy/Paste This)

🧠 Overview

This project uses a deep learning model (InceptionV3–based CNN) to classify cotton leaf diseases.
The application is deployed using Flask, where users can upload an image and get predictions instantly.

This project demonstrates:
	•	Image preprocessing and augmentation
	•	Transfer learning with a pre-trained CNN
	•	Model training and evaluation
	•	Flask-based web deployment
	•	Real-time image classification workflow

⸻

🚀 Tech Stack

Machine Learning / Deep Learning
	•	TensorFlow / Keras
	•	InceptionV3 (Transfer Learning)
	•	NumPy

Backend
	•	Python
	•	Flask
	•	OpenCV

Tools
	•	VS Code
	•	Git & GitHub
	•	Google Colab / Jupyter Notebook for training

⸻

📁 Project Structure

Cotton-Disease-Prediction/
│
├── app.py                     # Flask backend
├── incep.h5 (Not included in repo)  
│
├── model_predict/             # Prediction logic
│   ├── model.py
│   ├── img_path.py
│   └── pred_class.py
│
├── class_dict.py              # Label mapping
│
├── templates/                 # HTML files
│   └── index.html
│
└── static/                    # CSS, JS, Images


⸻

🎯 Model File (Important)

The trained model file (incep.h5) is NOT included in this repository because:
	1.	GitHub does not support large binary files (>100MB)
	2.	Model files are usually stored externally (Drive/S3)
	3.	This keeps the repository lightweight

👉 Download the trained model from here:

https://drive.google.com/file/d/1R_0u7eGQGP1iIYM1QQQ5CpEIZDX46na-/view

After downloading, place it in the project root like this:

Cotton-Disease-Prediction/
    app.py
    incep.h5    ← place here


⸻

🔧 How to Run the Project

1. Install Dependencies

pip install tensorflow opencv-python numpy flask pillow

2. Run Flask App

python app.py

3. Open in Browser

http://127.0.0.1:5000/

Upload an image → get predictions.

⸻

🧪 Model Training (Summary)

The model uses transfer learning based on Google’s InceptionV3:
	•	Input images resized to 299×299 × 3
	•	Frozen base layers of InceptionV3
	•	Added custom Dense layers for classification
	•	Trained on cotton leaf disease dataset
	•	Optimizer: Adam
	•	Loss: Categorical Crossentropy

Training script summary:

base = InceptionV3(weights='imagenet', include_top=False)
x = Flatten()(base.output)
x = Dense(256, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base.input, outputs=x)
model.save("incep.h5")


⸻

📝 Features

✔ Predicts cotton leaf disease from images
✔ Clean Flask interface
✔ Transfer learning boosts accuracy
✔ Easy to deploy and extend

⸻

🌐 Future Enhancements
	•	Deploy on AWS/GCP
	•	Add more crop diseases
	•	Convert model to TensorFlow Lite for mobile

⸻

🙋‍♂️ Author

Your Name
Machine Learning / Python Developer
GitHub: your-github-profile

⸻

🔥 This README looks professional and interviewer-ready.

If you want:
	•	A project explanation in STAR format
	•	A walkthrough for HR / Technical interviewer
	•	A portfolio version
	•	A YouTube video script for your project

Just say “Prepare interview explanation” and I’ll do it.
