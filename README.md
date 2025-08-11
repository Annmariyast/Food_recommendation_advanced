# Food Recommendation Advanced (Healthy Food Intake Monitor)

## Overview
**Food Recommendation Advanced** is a Python-based application designed to support healthier eating by offering personalized food suggestions based on dietary restrictions, health conditions, or user preferences.  
It leverages:
- A **structured dataset** mapping diseases to recommended foods.
- **Natural Language Processing (NLP)** to understand free-text input.
- **Voice-to-Text** input so users can speak their requests instead of typing.
- A simple recommendation engine to provide relevant suggestions.


## Features
- **Disease-specific healthy food recommendations.**
- **Voice-to-Text input** for hands-free usage.
- **Text-based input** (free-form natural language queries).
- **Image classification** for food recognition (CNN model).
- CSV-based disease-to-food mapping.
- Modular design for easy extension.

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/Annmariyast/Food_recommendation_advanced.git
   cd Food_recommendation_advanced


## NLP Model for Food Extraction
1. spaCy (Rule-based + Named Entity Recognition)
Custom pattern matcher for identifying food items in free-text input.
2. Voice-to-Text Model
SpeechRecognition library (using Google Web Speech API or compatible engine)
Converts spoken food names into text before NLP processing.
3. Rule-Based Recommendation Engine
Dataset-driven filtering (no machine learning model)
Matches extracted food items against disease-specific “safe” and “unsafe” lists in your CSV dataset.
4. Streamlit Front-End Logic
Handles user input, runs the NLP/voice pipeline, and displays results.

So technically, this system doesn’t train deep learning models — instead, it uses pre-trained NLP and speech recognition models combined with a custom rule-based recommendation engine.
