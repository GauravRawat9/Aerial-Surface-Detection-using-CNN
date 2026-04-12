# Aerial-Surface-Detection-using-CNN

A deep learning project that classifies aerial and satellite imagery into 45 scene categories using a custom-built Convolutional Neural Network trained on the NWPU-RESISC45 dataset — one of the most comprehensive remote sensing scene classification benchmarks available.

🎯 Project Overview
This project builds, trains, and fine-tunes a CNN from scratch to recognize land-use and land-cover scenes from aerial imagery. It includes a polished Streamlit web app for real-time inference, letting users upload any satellite image and get instant scene predictions with confidence scores.
Achieved ~71% test accuracy on 45 classes with a custom architecture — without using any pretrained backbone like ResNet or VGG.

📁 Project Structure
```
Aerial surface detection using CNN/
├── app.py                          # Streamlit UI
├── requirements.txt
└── base CNN -71 % accuracy/
    ├── cnn_resisc45.keras          # Trained base CNN
    ├── idx_to_class.json           # Index → class label map
    ├── class_indices.json          # Class label → index map
    ├── model_metadata.json         # Accuracy, epochs, config
    ├── training_history.json       # Loss/accuracy curves
    └── fine tuned/
        ├── cnn_resisc45_finetuned_best.keras   # Best fine-tuned checkpoint
        ├── cnn_resisc45_finetuned.keras        # Final fine-tuned model
        ├── finetune_log.csv                    # Per-epoch log
        └── training_history_finetuned.json     # Combined history
```
🗂️ Dataset — NWPU-RESISC45
```
Property             Value
Classes              45 scene categories
Images per class     700
Total images         31,500
Original resolution  256 × 256 px
Input to model       128 × 128 px
Source               Kaggle — NWPU-RESISC45
```
All 45 classes: airplane, airport, baseball diamond, basketball court, beach, bridge, chaparral, church, circular farmland, cloud, commercial area, dense residential, desert, forest, freeway, golf course, ground track field, harbor, industrial area, intersection, island, lake, meadow, medium residential, mobile home park, mountain, overpass, palace, parking lot, railway, railway station, rectangular farmland, river, roundabout, runway, sea ice, ship, snowberg, sparse residential, stadium, storage tank, tennis court, terrace, thermal power station, wetland.

🧠 Model Architecture
Base CNN
A custom Sequential CNN with 4 convolutional blocks followed by a dense classifier head.
```
Input (128×128×3)
│
├── Conv2D(32)  → BatchNorm → MaxPool
├── Conv2D(64)  → BatchNorm → MaxPool
├── Conv2D(128) → BatchNorm → MaxPool
├── Conv2D(256) → BatchNorm → MaxPool
│
├── Flatten
├── Dense(512) → Dropout(0.25)
├── Dense(256) → Dropout(0.25)
└── Dense(45, softmax)
```

```
Parameter           Value
Optimizer           Adam
Loss                Categorical Crossentropy
Precision           Mixed float16
Epochs              50
Batch size          32
```
Fine-tuning Strategy

1. Froze the first 2 convolutional blocks (low-level edge/texture detectors)
2. Kept BatchNorm layers frozen to preserve learned statistics
3. Unfroze Conv blocks 3 & 4 + dense head
4. Recompiled with learning rate 1e-4 (10× lower than base)
5. Stronger augmentation: rotation=45°, vertical_flip=True, fill_mode=reflect
6. Callbacks: EarlyStopping(patience=8), ReduceLROnPlateau, ModelCheckpoint


🖼️ Data Augmentation
```
Stage                     Augmentations Applied
Base training             Rotation ±40°, zoom 0.3, width/height shift 0.2, brightness [0.7–1.3], horizontal flip
Fine-tuning               All above + rotation ±45°, zoom 0.35, shear 0.2, channel shift 20, vertical flip, fill=reflect
```
Vertical flip is valid for aerial imagery since satellite images have no natural "up" orientation.


🚀 Streamlit Web App
A dark-themed, production-grade UI built with custom CSS.
Features:

1. Upload any aerial/satellite image (JPG, PNG, WEBP, TIF)
2. Switch between Base CNN and Fine-tuned CNN from a dropdown
3. Top-K predictions with animated confidence bars (adjustable 3–10)
4. Confidence score, inference time, entropy-based certainty metric
5. Live image stats: dimensions, color mode, average RGB channels
6. Optional full 45-class probability distribution
7. Model metadata display (test accuracy, total epochs trained)

Run locally:
```bash
# Clone or download the project
cd "Aerial surface detection using CNN"

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

📦 Requirements
streamlit>=1.32.0
tensorflow>=2.13.0
numpy>=1.24.0
Pillow>=10.0.0

📊 Results
```
Model                                    Test Accuracy
Base CNN (50 epochs)                     ~71%
Fine-tuned CNN (best checkpoint)         Higher (EarlyStopping on val_accuracy)
```

🔬 Key Technical Decisions
```
Decision                                    Reasoning
Custom CNN over transfer learning           Demonstrates architecture understanding from scratch
Mixed float16 precision                     Faster training and reduced memory on GPU
BatchNorm frozen during fine-tuning         Prevents degradation of learned stats with small LR
reflect fill mode                           Better than zero-padding for natural aerial textures
Entropy as certainty metric                 More meaningful than raw confidence for 45-class problems
```

📌 Future Improvements

 Experiment with EfficientNet / ResNet backbone via transfer learning
 Grad-CAM visualizations to highlight discriminative regions
 Ensemble base + fine-tuned model predictions
 Deploy on Streamlit Cloud or Hugging Face Spaces
 Add confusion matrix and per-class accuracy in the UI


Dataset: https://www.kaggle.com/datasets/aqibrehmanpirzada/nwpuresisc45
