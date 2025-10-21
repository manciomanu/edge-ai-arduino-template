# edge-ai-arduino-template
# Edge AI on Arduino — Template

This repository contains a simple template for running a tiny Machine Learning inference pipeline on a microcontroller (Arduino). It is part of my undergraduate thesis and serves as a starting point for future embedded AI experiments.

## 📌 What this project demonstrates
- A tiny inference flow (Python → exported weights → Arduino firmware)
- How to simulate a simple model and integrate it on a microcontroller
- The core idea of **Edge AI**: low-latency inference without cloud dependency

## 📁 Repository structure
edge-ai-arduino-template/
│
├── python/
│ └── train_and_export.py # Generates quantized weights
│
└── arduino/
└── edge_ai_demo.ino # Simple Arduino inference loop

## 🚀 How to use
1. Run the Python script to generate model weights  
2. Copy the exported header to the Arduino folder (if needed)  
3. Upload the `.ino` file to your board and check the serial output

## 🧩 Next steps (for future versions)
- Replace the placeholder model with a real TinyML approach (e.g., TFLite Micro)
- Collect and train with real sensor data (accelerometer, temperature, etc.)
- Benchmark RAM, Flash and latency

---

📌 *Author:* **Manoela Mancio**  
🎓 Undergraduate Thesis — Edge AI on Microcontrollers
