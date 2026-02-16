# AIConceptsNET 🤖
A comprehensive collection of C# Console Applications demonstrating Artificial Intelligence and Machine Learning concepts using .NET 8 and ML.NET v5.0.
Code AI, ML, Neural Networks, and more in C#!

# 🚀 Overview
This repository serves as a practical laboratory for exploring various AI paradigms within the .NET ecosystem. From traditional statistical regression to modern Generative AI, each project is a standalone implementation designed to show how C# can be used for high-performance AI tasks without leaving the Microsoft stack.

# 🛠 Tech Stack
Runtime: .NET 8

Primary Library: ML.NET v5.0

Language: C# 12

Integrations: ONNX Runtime, Microsoft.ML.GenAI, and TorchSharp.

# 📁 Project Catalog

| Project | Concept | Description |
| :--- | :--- | :--- |
| AnomalyDetection | Time-Series | Identifies spikes and change points in sequential data (e.g., fraud detection). |
| Clustering | Unsupervised | Groups data points (e.g., customer segments) using K-Means clustering. |
| ForecastingAl | Time-Series | Predicts future values based on historical patterns (SSA/Regression). |
| ImageClassification | Deep Learning | Uses Transfer Learning (TensorFlow/TorchSharp) to categorize images. |
| MachineLearningExample | General ML | A baseline implementation of a standard ML.NET training pipeline. |
| MazeExample | Reinforcement | A Q-Learning implementation for pathfinding in a grid environment. |
| NeuralNetwork | Deep Learning | Explores custom layers and ONNX model integration for complex data. |
| Recommendation | Collaborative Filtering | Suggests items based on user-item interaction scores (Matrix Factorization). |
| Regression | Supervised | Predicts continuous numerical values (e.g., house prices or stock trends). |
| SentimentAnalysis | NLP | Classifies text as positive, negative, or neutral using binary classification. |
| TextGeneration | Generative AI | Local LLM implementation (Phi-3/4) for generating text via prompts. |


# 🔍 Deep Dives🧩

## MazeExample (Reinforcement Learning)

The MazeExample demonstrates Q-Learning, a model-free reinforcement learning algorithm. Unlike the other projects that use pre-existing datasets, this agent learns by interacting with its environment.

### **The Logic**
*   **States:** The current cell index of the agent.
*   **Actions:** Moves (Up, Down, Left, Right).
*   **Rewards:** Points for reaching the goal, penalties for hitting walls or taking too many steps.
*   **The Equation:** The agent updates its "Quality" ($Q$) matrix using the **Bellman Equation**:

$$Q(s, a) = Q(s, a) + \alpha [R + \gamma \max Q(s', a') - Q(s, a)]$$


# ✍️ TextGeneration (Generative AI)
This project leverages the modern ML.NET 5.0 capabilities to run Small Language Models (SLMs) locally.

### **Features**
*   **Model Support:** Configured for **Phi-3** or **Phi-4** in **ONNX** format.
*   **Streaming Output:** Implements `IAsyncEnumerable` to display text in real-time as the model generates tokens.
*   **Tokenization:** Utilizes the `Microsoft.ML.Tokenizers` library for high-speed local inference.


# 🧠 Key Learnings

### **Core Capabilities**
*   **Data Pipelines:** Using `IDataView` for memory-efficient data loading.
*   **Model Evaluation:** Measuring success via **R-Squared**, **F1-Score**, and **RMSE**.
*   **Local GenAI:** Running LLMs locally to ensure **data privacy** and zero latency costs.
