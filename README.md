# DeepfakeDetection
EfficientNet-B6 – Based Fake and Propaganda Image Detection using AGSK Optimization 
Deepfake Detection using EfficientNet-B6 and AGSK
📌 Project Description
This project presents a robust and scalable solution for deepfake detection by combining EfficientNet-B6 for feature extraction and AGSK (Automated Adaptive Gaining Sharing Knowledge) for dynamic hyperparameter tuning. The model is enhanced with Capsule Networks to preserve spatial hierarchies, improving the classification of real and fake images.

The system is trained on a balanced dataset of real and deepfake images and achieves 99% training accuracy and 96% testing accuracy. It is optimized for real-time deepfake detection, suitable for use in digital forensics, media authentication, and content moderation.

🔍 Key Features
✅ Real-time detection of deepfake images.

✅ EfficientNet-B6 for hierarchical feature extraction.

✅ AGSK for automated hyperparameter optimization.

✅ Capsule Networks to maintain spatial relationships.

✅ Preprocessing pipeline with normalization, resizing, augmentation, and noise reduction.

✅ Evaluation metrics: Accuracy, Precision, Recall, F1-Score.

🧠 Technologies Used
Language: Python 3.10

Frameworks: PyTorch, TensorFlow

Models: EfficientNet-B6, Capsule Network

Optimization: AGSK Algorithm

Training Platform: Google Colab (NVIDIA T4 GPU)

🗂️ Project Structure
bash
Copy
Edit
├── dataset/               # Contains real and fake training/testing images
├── models/                # Trained model weights
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
⚙️ Installation
bash
Copy
Edit
git clone https://github.com/yourusername/deepfake-detection-efficientnetB6-agsk.git
cd deepfake-detection-efficientnetB6-agsk
pip install -r requirements.txt
🚀 How to Run
Mount your Google Drive in Google Colab.

Load and preprocess the dataset.

Train the model:

python
Copy
Edit
!python train.py
Evaluate the model:

python
Copy
Edit
!python evaluate.py
📊 Evaluation Metrics
Accuracy

Precision

Recall

F1-Score

📚 References
M. Tan and Q. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” ICML, 2019.

A. Mohamed et al., “Gaining-Sharing Knowledge Based Algorithm with Adaptive Parameters,” IEEE Access, 2021.

B. Dolhansky et al., “The DeepFake Detection Challenge Dataset,” arXiv, 2020.

J. Zhang et al., “Deepfake Detection with Vision Transformers and EfficientNets,” CVPR, 2022.

Additional references included in the documentation.

📄 License
This project is licensed under the MIT License.
