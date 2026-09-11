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
ALLOWED_NODES = ["Disease", "Medication", "Procedure", "Symptom", "BodyPart", "Test"]

if "nx_graph" not in st.session_state:
    st.session_state.nx_graph = nx.DiGraph()

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )

def ingest_data_to_graph(file_path: str, record_limit: int = 100):
    llm = get_llm()

    df = pd.read_csv(file_path)
    records = df['transcription'].dropna().head(record_limit).tolist()
    documents = [Document(page_content=text) for text in records]

    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES
    )

    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    for doc in graph_documents:
        for node in doc.nodes:
            st.session_state.nx_graph.add_node(node.id, type=node.type)
        for rel in doc.relationships:
            st.session_state.nx_graph.add_edge(
                rel.source.id, 
                rel.target.id, 
                relation=rel.type
            )

st.set_page_config(page_title="PulseNet Intelligence Engine", layout="wide")
st.title("🩺 PulseNet — Clinical Knowledge Graph Intelligence Engine")

try:
    llm = get_llm()
except Exception as e:
    st.error(f"Failed to connect to Gemini API: {e}")
    st.stop()

with st.sidebar:
    st.header("⚙️ Data Pipeline")
    uploaded_file = st.file_uploader("Upload Medical Transcripts CSV", type=["csv"])
    limit = st.number_input("Records to process", min_value=1, max_value=5000, value=10)
    
    if st.button("Construct Knowledge Network"):
        if uploaded_file is not None:
            with st.spinner("Processing clinical records..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                ingest_data_to_graph(tmp_path, record_limit=limit)
                st.success(f"Network updated. Nodes: {st.session_state.nx_graph.number_of_nodes()} | Edges: {st.session_state.nx_graph.number_of_edges()}")
        else:
            st.warning("Please upload a CSV file.")

tab1, tab2, tab3 = st.tabs(["💬 Clinical Intelligence Q&A", "🕸️ Interactive Network Explorer", "🔍 Knowledge Matrix Data"])

with tab1:
    st.subheader("Traverse Knowledge Network via Natural Language")
    user_query = st.text_input(
        "Ask a clinical question:",
        key="nl_query"
    )

    if st.button("Run Intelligence Search") and user_query:
        if st.session_state.nx_graph.number_of_nodes() == 0:
            st.error("Knowledge base is empty. Upload data in sidebar first.")
        else:
            with st.spinner("Evaluating relational paths..."):
                try:
                    triples = []
                    for u, v, data in st.session_state.nx_graph.edges(data=True):
                        rel = data.get("relation", "RELATED_TO")
                        triples.append(f"({u}) -[{rel}]-> ({v})")
                    
                    context = "\n".join(triples[:300])

                    prompt = ChatPromptTemplate.from_template(
                        "You are PulseNet, an advanced clinical reasoning system. Synthesize a response based strictly on these verified clinical relationships.\n\n"
                        "Knowledge Base Context:\n{context}\n\n"
                        "Clinical Query: {question}\n\n"
                        "Synthesis:"
                    )
                    
                    chain = prompt | llm
                    response = chain.invoke({"context": context, "question": user_query})

                    st.markdown("### Synthesized Clinical Answer")
                    st.write(response.content)

                    with st.expander("Retrieved Network Context Triples"):
                        st.code(context, language="text")
                except Exception as e:
                    st.error(f"Execution failure: {e}")

with tab2:
    st.subheader("Sub-Network Visualizer")
    node_limit = st.slider("Max render edges", 10, 200, 40)

    if st.button("Render Visual Network"):
        if st.session_state.nx_graph.number_of_nodes() == 0:
            st.warning("No data loaded in memory.")
        else:
            try:
                net = Network(height="550px", width="100%", directed=True)

                edges = list(st.session_state.nx_graph.edges(data=True))[:node_limit]
                for source, target, data in edges:
                    relation = data.get("relation", "RELATED_TO")

                    net.add_node(source, label=source, title=source)
                    net.add_node(target, label=target, title=target)
                    net.add_edge(source, target, title=relation, label=relation)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
                    net.save_graph(tmp_html.name)
                    components.html(open(tmp_html.name, 'r').read(), height=570)
            except Exception as e:
                st.error(f"Visualization rendering error: {e}")

with tab3:
    st.subheader("Knowledge Graph Relational Edge List")
    if st.session_state.nx_graph.number_of_nodes() == 0:
        st.info("No active graph edges available.")
    else:
        table_data = []
        for u, v, data in st.session_state.nx_graph.edges(data=True):
            table_data.append({
                "Subject (Source)": u,
                "Predicate (Relation)": data.get("relation", "RELATED_TO"),
                "Object (Target)": v
            })
        df_edges = pd.DataFrame(table_data)
        st.dataframe(df_edges, use_container_width=True)