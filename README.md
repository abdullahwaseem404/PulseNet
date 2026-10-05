# 🩺 PulseNet — Clinical Knowledge Graph Intelligence Engine

PulseNet is an **AI-powered clinical knowledge graph application** that transforms unstructured medical transcripts into a structured network of clinical entities and relationships.

The system uses **Google Gemini, LangChain, NetworkX, Pandas, Streamlit, and PyVis** to extract clinical knowledge and provide an interactive environment for exploring relationships between diseases, symptoms, medications, procedures, body parts, tests, dosage, severity, and healthcare providers.

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
  * Dosage
  * Severity
  * Providers

* 💬 Ask clinical questions using natural language

* 🔍 Generate responses using retrieved graph relationships

* 🕸️ Interactive clinical network visualization using **PyVis**

* 📊 Explore extracted knowledge in a structured matrix

* ⚡ Fast and interactive **Streamlit** web interface

* 🧾 Inspect graph triples used for generated answers

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
                 │       Pandas         │
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
                 │     NetworkX Graph   │
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

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🔬 Example Workflow

```text
Upload Medical CSV
        ↓
Select Records to Process
        ↓
Construct Knowledge Graph
        ↓
Gemini Extracts Clinical Entities
        ↓
LangChain Builds Relationships
        ↓
NetworkX Stores the Graph
        ↓
 ┌────────────────┬──────────────────┬─────────────────┐
 │                │                  │                 │
 ▼                ▼                  ▼                 │
Clinical Q&A   Network Explorer   Knowledge Matrix   │
 │                │                  │                 │
 └────────────────┴──────────────────┴─────────────────┘
```

---

## 🧩 Technologies Used

* **Python** — Core programming language
* **Pandas** — Medical transcript processing
* **Google Gemini** — Clinical entity and relationship extraction
* **LangChain** — LLM graph transformation and prompting
* **NetworkX** — Graph construction and relationship storage
* **PyVis** — Interactive graph visualization
* **Streamlit** — Web application interface

---

## ⚠️ Important Note

PulseNet is an **educational and research-oriented AI application**.

Generated outputs should **not be considered medical advice, diagnosis, or treatment recommendations**. Clinical information should always be reviewed and verified by qualified healthcare professionals and trusted medical sources.

---
