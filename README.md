# AI-Driven Emotion Detection & Personalized Learning Support Platform

An emotion-aware learning support application that analyzes a learner's free-text study challenge, estimates learning-related emotional context, and provides personalized academic guidance.

## Features

- Five emotion classes: Bored, Confident, Confused, Curious, Frustrated
- BiLSTM-style sequence/context inference for instant demo use
- BERT-family transformer emotion inference with rule-based learning-context enhancement
- Mixed-emotion breakdown
- Side-by-side model comparison
- Optional Gemini-generated personalized learning guidance
- Local fallback guidance when Gemini is disabled
- CSV interaction logging
- Streamlit analytics dashboard
- Responsible AI notice

## Project Architecture

Student Input  
→ Text Processing  
→ BiLSTM-style Sequence Scoring + BERT  
→ Rule-Based Keyword Enhancement  
→ Mixed Emotion Detection  
→ Final Emotion Prediction  
→ Gemini AI / Local Guidance  
→ Tips, Next Steps & Encouragement  
→ CSV Logging  
→ Analytics Dashboard

## Important Implementation Note

The repository is designed to run immediately for demonstration. The application uses a bidirectional sequence/context scoring component on the BiLSTM side when no trained checkpoint is available. A real PyTorch `BiLSTMClassifier` architecture and training pipeline are included in `train_bilstm.py`. Meaningful deep-learning training requires a larger, properly labeled dataset.

The transformer side attempts to load a pretrained Hugging Face emotion classifier and maps generic emotions into the project's five learning-emotion classes. Rule-based enhancement adds learning-context signals.

## Installation

1. Clone the repository.
2. Open the project folder in VS Code.
3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate it on Windows:

```bash
venv\Scripts\activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Run the application:

```bash
streamlit run app.py
```

## Gemini Setup

Gemini is optional. The app works with local fallback guidance when Gemini is disabled.

For Gemini, set an environment variable:

```bash
set GEMINI_API_KEY=YOUR_KEY
```

Never upload your real API key to GitHub.

## Train the PyTorch BiLSTM Reference Model

A small sample dataset is included only to demonstrate the pipeline:

```bash
python train_bilstm.py
```

For meaningful model performance, replace `data/training_data.csv` with a larger labeled dataset containing `text` and `label` columns.

## Technologies

- Python
- PyTorch
- Transformer Models
- BiLSTM / LSTM
- Streamlit
- NumPy
- Pandas
- Gemini AI
- Prompt Engineering
- Responsible AI

## Use Cases

### Educators, TAs & Academic Support Teams
Academic support staff can analyze student messages, review emotion signals and compare model predictions before providing targeted support.

### Learners & Self-Study
Students can describe a learning challenge in plain language and receive emotion-aware study strategies, next steps and encouragement.

## Responsible AI

This system provides academic learning support. Emotion predictions are contextual estimates and are not medical or psychological diagnoses. Model predictions can be imperfect and should not be used as the sole basis for high-stakes decisions.

## Example

Input:

> I tried recursion many times but I still don't understand the base case.

Possible result:

- Final Emotion: Frustrated
- Mixed Emotions: Frustrated + Confused
- Personalized guidance focused on isolating the unclear step and practicing an easier example.
