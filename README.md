# 🩺 PulseNet — Clinical Knowledge Graph Intelligence Engine

PulseNet is an **AI-powered clinical knowledge graph application** that converts unstructured medical transcripts into a structured network of clinical entities and relationships.

The system uses **Google Gemini, LangChain, NetworkX, and PyVis** to extract medical knowledge and provide an interactive interface for exploring relationships between diseases, symptoms, medications, procedures, body parts, and diagnostic tests.

---

## 🚀 Features

* 📄 Upload medical transcript datasets in CSV format
* 🤖 Extract clinical entities and relationships using **Google Gemini**
* 🧠 Build a dynamic **Clinical Knowledge Graph**
* 🔗 Identify relationships between:

  * Diseases
  * Symptoms
  * Medications
  * Procedures
  * Body Parts
  * Medical Tests
* 💬 Ask clinical questions using natural language
* 🔍 Generate answers using retrieved knowledge-graph relationships
* 🕸️ Interactive network visualization using **PyVis**
* 📊 Explore extracted relationships in a structured data table
* ⚡ Streamlit-based interactive web interface

---

## 🏗️ Architecture

```text
                 ┌──────────────────────┐
                 │   Medical CSV Data   │
                 │     Transcripts      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      Pandas          │
                 │   Data Processing    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Google Gemini      │
                 │    + LangChain       │
                 │ Graph Transformation │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   NetworkX Graph     │
                 │ Nodes + Relationships│
                 └──────────┬───────────┘
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
        ┌────────────┐ ┌──────────┐ ┌────────────┐
        │ Clinical   │ │ Network  │ │ Knowledge  │
        │ Q&A        │ │ Explorer │ │ Matrix     │
        └────────────┘ └──────────┘ └────────────┘
```

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/abdullahwaseem404/PulseNet.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔬 Example Workflow

```text
Upload CSV
    ↓
Select number of records
    ↓
Construct Knowledge Network
    ↓
Gemini extracts clinical entities
    ↓
LangChain creates graph relationships
    ↓
NetworkX stores the graph
    ↓
 ┌───────────────┬──────────────────┬─────────────────┐
 │               │                  │                 │
 ▼               ▼                  ▼                 │
Clinical Q&A   Network Explorer   Knowledge Matrix    │
 │               │                  │                 │
 └───────────────┴──────────────────┴─────────────────┘
```

---

## ⚠️ Important Note

PulseNet is an **educational and research-oriented AI application**.

The generated answers should **not be considered medical advice, diagnosis, or treatment recommendations**. Clinical information should always be verified using qualified healthcare professionals and trusted medical sources.

---
