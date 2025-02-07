# AIstronauts: Space Agents on a Mission 🚀

Welcome to **AIstronauts**, an AI-powered system designed to revolutionize space operations by automating mission planning and ground station workflows. This project combines **Reinforcement Learning (RL)** and **Natural Language Processing (NLP)** to create intelligent agents for space missions.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Contributing](#contributing)
8. [License](#license)

---

## **Project Overview**
AIstronauts is a **Space Operations Management System** that integrates two main components:
1. **Mission Planning System**: Uses Reinforcement Learning (RL) to optimize mission plans for space operations.
2. **Ground Station RPA System**: Automates command processing and workflow execution using Natural Language Processing (NLP).

This project is built for the **AI for Connectivity Hackathon II: Building Resilient Networks** hosted by **LabLab AI**.

---

## **Features**
- **Mission Planning**:
  - Simulates mission environments using OpenAI Gym.
  - Trains a Deep Q-Network (DQN) agent to optimize mission plans.
  - Generates mission timelines and visualizes training progress.
- **Ground Station RPA**:
  - Processes natural language commands using DistilBERT (NLP).
  - Executes predefined workflows for tasks like telemetry collection and system checks.
  - Provides real-time analytics and metrics for system performance.
- **Interactive Interface**:
  - Built using **Streamlit** for an intuitive and user-friendly experience.
  - Visualizes data using **Plotly** for mission timelines, training progress, and workflow execution.

---

## **Technology Stack**
- **Programming Language**: Python
- **Frameworks/Libraries**:
  - **Streamlit**: For building the interactive web interface.
  - **PyTorch**: For implementing the Deep Q-Network (DQN).
  - **Transformers (Hugging Face)**: For NLP tasks like sentiment analysis.
  - **Plotly**: For data visualization.
  - **Gymnasium (OpenAI Gym)**: For simulating the mission environment.
  - **Pandas/Numpy**: For data manipulation and analysis.
- **Pre-trained Models**:
  - **DistilBERT**: Used for sentiment analysis and command classification.

---

## **Installation**
To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sajjadahmad-dev/AIstronauts-Space-Agents-Hackathon
   cd AIstronauts-Space-Agents-Hackathon
