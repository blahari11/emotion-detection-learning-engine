# Project Explanation for Viva

## One-line explanation
Our project detects a learner's study-related emotion from free-text input and provides personalized academic guidance based on the detected emotional context.

## Problem
Teachers and students spend time manually interpreting study difficulties. A normal chatbot may answer the topic but may not adapt the response tone or strategy to whether the learner is confused, bored, curious, confident, or frustrated.

## Solution
The platform analyzes student text using two model paths, enhances predictions with learning-context keywords, identifies mixed emotions, and generates personalized learning support.

## Why BiLSTM?
BiLSTM processes sequential information in forward and backward directions. This helps model word-order and surrounding context.

## Why BERT?
BERT-family transformer models use attention-based contextual representations and can understand sentence context better than simple keyword matching.

## Why rule-based enhancement?
Learning phrases such as "still can't", "don't understand", "what if", and "clear now" carry strong domain-specific signals. Rules enhance model output for the educational context.

## Why Gemini?
Gemini converts the detected emotion and study problem into natural, personalized guidance, learning strategies, next steps, and encouragement.

## Why CSV?
CSV logging creates a simple interaction history that can be analyzed and later used for continuous improvement.

## Responsible AI
The system does not diagnose mental-health conditions. Emotion labels are contextual learning signals. The user should be informed that predictions can be imperfect.

## Viva Questions

1. What is the objective of the project?
To provide emotion-aware personalized learning support from student text.

2. What are the emotion classes?
Bored, Confident, Confused, Curious, and Frustrated.

3. What is BiLSTM?
A bidirectional Long Short-Term Memory network that processes sequence context in both forward and backward directions.

4. What is BERT?
A transformer-based language representation model that uses contextual attention.

5. Why compare two models?
To improve transparency and show how different model approaches interpret the same input.

6. What is mixed-emotion detection?
Displaying the strongest two emotional signals when the text contains more than one relevant emotion.

7. What is prompt engineering?
Designing structured instructions for a generative AI model so it returns useful, safe, and consistent output.

8. What is the role of Streamlit?
It provides the web interface, controls, results, charts, and analytics dashboard.

9. How is data stored?
Interactions are appended to a CSV file.

10. What is a limitation?
Prediction accuracy depends on training data, pretrained model behavior, domain mapping, and the ambiguity of human language.
