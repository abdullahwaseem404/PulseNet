import os
import tempfile
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from dotenv import load_dotenv
from pyvis.network import Network

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ALLOWED_NODES = [
    "Disease", "Symptom", "Medication", "Procedure", 
    "BodyPart", "Test", "Dosage", "Severity", "Provider"
]

if "nx_graph" not in st.session_state:
    st.session_state.nx_graph = nx.MultiDiGraph()

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )

def ingest_data_to_graph(file_path: str, record_limit: int = 3):
    llm = get_llm()
    df = pd.read_csv(file_path)
    
    text_col = 'transcription' if 'transcription' in df.columns else df.columns[0]
    
    records = df[text_col].dropna().head(record_limit).tolist()
    documents = [
        Document(page_content=str(text)[:1500], metadata={"source_id": idx}) 
        for idx, text in enumerate(records)
    ]

    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
        node_properties=["assertion", "temporal_context"]
    )

    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    for doc in graph_documents:
        for node in doc.nodes:
            st.session_state.nx_graph.add_node(
                node.id, 
                type=node.type, 
                properties=node.properties
            )
        for rel in doc.relationships:
            st.session_state.nx_graph.add_edge(
                rel.source.id, 
                rel.target.id, 
                relation=rel.type,
                properties=rel.properties
            )

st.set_page_config(page_title="PulseNet Intelligence Engine", layout="wide")
st.title("🩺 PulseNet — Temporal Clinical GraphRAG & Knowledge Intelligence Engine")

try:
    llm = get_llm()
except Exception as e:
    st.error(f"Failed to connect to Gemini API: {e}")
    st.stop()

with st.sidebar:
    st.header("⚙️ Fast Clinical ETL")
    uploaded_file = st.file_uploader("Upload Medical Transcripts CSV", type=["csv"])
    limit = st.number_input("Records to process (Keep low for speed)", min_value=1, max_value=50, value=3)
    
    if st.button("Construct Knowledge Graph"):
        if uploaded_file is not None:
            with st.spinner("Extracting clinical entities quickly..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    ingest_data_to_graph(tmp_path, record_limit=limit)
                    st.success(f"Done! Nodes: {st.session_state.nx_graph.number_of_nodes()} | Edges: {st.session_state.nx_graph.number_of_edges()}")
                except Exception as ex:
                    st.error(f"Extraction error: {ex}")
        else:
            st.warning("Please upload a CSV file.")

tab1, tab2, tab3 = st.tabs(["💬 Subgraph GraphRAG Q&A", "🕸️ Interactive Clinical Network", "🔍 Provenance & Knowledge Matrix"])

with tab1:
    st.subheader("GraphRAG Natural Language Clinical Reasoning")
    user_query = st.text_input(
        "Ask a clinical question (e.g., 'What treatments or medications are mentioned?'):",
        key="nl_query"
    )

    if st.button("Run Intelligence Search") and user_query:
        if st.session_state.nx_graph.number_of_nodes() == 0:
            st.error("Knowledge base is empty. Upload clinical data in the sidebar first (try setting records to 3).")
        else:
            with st.spinner("Analyzing graph relationships..."):
                try:
                    graph = st.session_state.nx_graph
                    relevant_triples = []
                    
                    for u, v, data in graph.edges(data=True):
                        rel = data.get("relation", "RELATED_TO")
                        relevant_triples.append(f"({u}) -[{rel}]-> ({v})")
                    
                    context = "\n".join(relevant_triples[:50])

                    prompt = ChatPromptTemplate.from_template(
                        "You are PulseNet, an advanced clinical reasoning system.\n\n"
                        "Retrieved Graph Context:\n{context}\n\n"
                        "Clinical Query: {question}\n\n"
                        "Synthesis:"
                    )
                    
                    chain = prompt | llm
                    response = chain.invoke({"context": context, "question": user_query})

                    st.markdown("### Synthesized Clinical Output")
                    st.write(response.content)

                    with st.expander("Inspected Graph Triples"):
                        st.code(context, language="text")
                except Exception as e:
                    st.error(f"Execution error: {e}")

with tab2:
    st.subheader("Interactive Sub-Network Visualizer")
    if st.button("Render Network View"):
        if st.session_state.nx_graph.number_of_nodes() == 0:
            st.warning("No data loaded in memory.")
        else:
            try:
                net = Network(height="500px", width="100%", directed=True, notebook=False)
                for source, target, data in list(st.session_state.nx_graph.edges(data=True))[:50]:
                    relation = data.get("relation", "RELATED_TO")
                    net.add_node(source, label=source)
                    net.add_node(target, label=target)
                    net.add_edge(source, target, title=relation, label=relation)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
                    net.save_graph(tmp_html.name)
                    components.html(open(tmp_html.name, 'r').read(), height=520)
            except Exception as e:
                st.error(f"Visualization error: {e}")

with tab3:
    st.subheader("Clinical Knowledge Matrix")
    if st.session_state.nx_graph.number_of_nodes() == 0:
        st.info("No active graph records available.")
    else:
        table_data = [
            {"Subject": u, "Predicate": data.get("relation", "RELATED_TO"), "Object": v}
            for u, v, data in st.session_state.nx_graph.edges(data=True)
        ]
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)