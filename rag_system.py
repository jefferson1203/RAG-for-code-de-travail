"""
Interface Streamlit pour le système RAG du Code du Travail
Ce script implémente un système RAG pour répondre aux questions sur le Code du Travail
en utilisant LangChain, ChromaDB et Gemini.
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional, List, Dict, Any
import asyncio
from langchain_core.documents import Document
import time
import pysqlite3
# Ajout du code d'initialisation SQLite
sys.modules['sqlite3'] = pysqlite3

# Configuration pour résoudre les problèmes de boucle d'événements
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Désactiver le watcher de Streamlit pour éviter les conflits avec PyTorch
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# Chargement des variables d'environnement
load_dotenv()

# Vérification des clés API requises
REQUIRED_ENV_VARS = {
    'GOOGLE_API_KEY': 'Google API Key pour Gemini',
    'LANGCHAIN_API_KEY': 'LangChain API Key pour le tracing'
}

missing_vars = [var for var, desc in REQUIRED_ENV_VARS.items() if not os.getenv(var)]
if missing_vars:
    error_msg = "ERREUR: Variables d'environnement manquantes:\n"
    for var in missing_vars:
        error_msg += f"- {var}: {REQUIRED_ENV_VARS[var]}\n"
    st.error(error_msg)
    st.stop()

# Configuration de LangChain
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Import des dépendances LangChain
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
    from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
except ImportError as e:
    st.error(f"ERREUR: Impossible de charger les dépendances requises: {str(e)}")
    st.info("Veuillez installer les dépendances avec: pip install -r requirements.txt")
    st.stop()

# Configuration de l'interface Streamlit
st.set_page_config(
    page_title="Assistant Juridique RAG (Code du Travail)",
    page_icon="⚖️",
    layout="wide"
)

# Titre et description
st.title("⚖️ Assistant Juridique")
st.markdown("""
Bienvenue dans votre outil d'aide à la décision juridique. Posez des questions sur le Code du Travail
et obtenez des réponses précises basées sur les documents officiels.
""")

    # Configuration du LLM
if "params" not in st.session_state:
    st.session_state.params = {
            "temperature": 0.2,
            "top_p": 0.8,
            "model": "gemini-2.0-flash"
        }

# Configuration des chemins
# PDF_PATH = "./data/data_50_page.pdf"
PDF_PATH = "./data/code_du_travail.pdf"
# PERSIST_DIRECTORY = "./chroma_db_code_travail"
PERSIST_DIRECTORY = "./chroma_db_codetravail"

# Vérification des chemins
if not os.path.exists(os.path.dirname(PDF_PATH)):
    st.error(f"ERREUR: Le dossier 'data' n'existe pas.")
    st.info("Veuillez créer un dossier 'data' et y placer votre fichier PDF.")
    st.stop()

# --- 1. Préparation des documents (Fonction cachée avec @st.cache_data) ---
@st.cache_data(show_spinner=False)
def load_documents() -> Optional[List[Any]]:
    """
    Charge le document PDF et retourne une liste de documents.
    Utilise le cache de Streamlit pour éviter de recharger le PDF à chaque fois.
    """
    print("Chargement du PDF...")
    if not os.path.exists(PDF_PATH):
        st.error(f"ERREUR: Le fichier PDF '{PDF_PATH}' n'a pas été trouvé.")
        return None

    try:
        loader = PyPDFLoader(PDF_PATH)
        _documents = loader.load()
        print(f"PDF chargé avec succès : {len(_documents)} pages")
        return _documents
    except Exception as e:
        st.error(f"ERREUR lors du chargement du PDF : {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def split_documents(_documents: List[Any]) -> Optional[List[Any]]:
    """
    Divise les documents en chunks plus petits pour le traitement.
    Utilise des patterns spécifiques au Code du Travail pour une meilleure segmentation.
    """
    print(" Division des documents...")
    if not _documents or len(_documents) == 0:
        st.error("Aucun document à diviser")
        return None

    try:
        # Vérification du contenu des documents
        for doc in _documents:
            if not hasattr(doc, 'page_content') or not doc.page_content:
                st.error("Document invalide : contenu manquant")
                return None

        text_splitters_patterns = [
            r"(Article\s+((?:L|R|D)\.\s*\d{3}-\d+(?:-\d+)?(?:-\d+)?)[\s\S]*?(?=Article\s+((?:L|R|D)\.\s*\d{3}-\d+(?:-\d+)?(?:-\d+)?)|Titre\s+|Chapitre\s+|Section\s+|$))",
            r"(Section\s+(?:\d+|unique|[A-Za-z\d\u00C0-\u00FF'-]+)\s*:\s*.+?)(?=\n(?:Titre\s|Chapitre\s|Section\s|Article\s|$))",
            r"(Chapitre\s+(?:[IVXLCDM]+(?:er|ème)?|unique|[A-Za-z\d\u00C0-\u00FF'-]+)\s*:\s*.+?)(?=\n(?:Titre\s|Chapitre\s|Section\s|Article\s|$))",
            r"(Titre\s+(?:[IVXLCDM]+(?:er|ème)?|[A-Za-z\d\u00C0-\u00FF'-]+)\s*:\s*.+?)(?=\n(?:Titre\s|Chapitre\s|Section\s|Article\s|$))",
            "\n\n", "\n", " ", ""
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  
            chunk_overlap=100,  
            separators=text_splitters_patterns,
            length_function=len,
            is_separator_regex=True
        )

        # Division des documents avec gestion d'erreur
        chunks = []
        for doc in _documents:
            try:
                doc_chunks = text_splitter.split_documents([doc])
                if doc_chunks:
                    chunks.extend(doc_chunks)
            except Exception as e:
                print(f"Attention : Erreur lors de la division d'un document : {str(e)}")
                continue

        if not chunks:
            st.error("Aucun chunk n'a pu être créé à partir des documents")
            return None

        print(f" Documents divisés en {len(chunks)} chunks")
        return chunks

    except Exception as e:
        st.error(f"ERREUR lors de la division des documents : {str(e)}")
        return None

# --- 2. Création/Chargement de ChromaDB ---
@st.cache_resource
def get_embedding_model():
    """
    Initialise et retourne le modèle d'embeddings.
    Utilise le cache de Streamlit pour éviter de recharger le modèle à chaque fois.
    """
    print("🤖 Initialisation du modèle d'embeddings...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(" Modèle d'embeddings initialisé")
        return embedding_model
    except Exception as e:
        st.error(f"ERREUR lors de l'initialisation du modèle d'embeddings : {str(e)}")
        return None

@st.cache_resource
def get_vectorstore():
    """
    Crée ou charge la base de données vectorielle.
    Utilise le cache de Streamlit pour éviter de recréer la base à chaque fois.
    """
    print(" Configuration de la base de données vectorielle...")
    try:
        embedding_model = get_embedding_model()
        if embedding_model is None:
            return None

        # Vérification et nettoyage du dossier de persistance si nécessaire
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                # Test de chargement de la base
                test_vectorstore = Chroma(
                    persist_directory=PERSIST_DIRECTORY,
                    embedding_function=embedding_model,
                    collection_metadata={"hnsw:space": "cosine"}
                )
                # Si on arrive ici, la base est valide
                print(" Chargement de la base existante...")
                return test_vectorstore
            except Exception as e:
                print(f" Base de données corrompue détectée: {str(e)}")
                print("Nettoyage et recréation de la base...")
                import shutil
                shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
        else:
            print(" Création d'une nouvelle base...")

        _documents = load_documents()
        if _documents is None:
            return None

        chunks = split_documents(_documents)
        if chunks is None:
            return None

        vectorstore = Chroma.from_documents(
            chunks,
            embedding=embedding_model,
            persist_directory=PERSIST_DIRECTORY,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(" Nouvelle base de données vectorielle créée")
        return vectorstore

    except Exception as e:
        st.error(f"ERREUR lors de la configuration de la base vectorielle : {str(e)}")
        return None

# --- 3. Initialisation du LLM et de la chaîne RAG ---
@st.cache_resource
def get_llm(_params: Dict[str, Any]) -> Optional[Any]:
    """
    Initialise et retourne le modèle de langage.
    Utilise le cache de Streamlit pour éviter de recharger le modèle à chaque fois.
    """
    print(" Initialisation du modèle de langage...")
    try:
        # Validation et conversion de top_p
        top_p = float(_params["top_p"])
        if not 0.0 <= top_p <= 1.0:
            top_p = 0.8  

        llm = ChatGoogleGenerativeAI(
            model=_params["model"],
            temperature=float(_params["temperature"]),
            max_output_tokens=2048,
            top_p=top_p
        )
        print(" Modèle de langage initialisé")
        return llm
    except Exception as e:
        st.error(f"ERREUR lors de l'initialisation du LLM : {str(e)}")
        return None

@st.cache_resource
def initialize_rag_system():
    """
    Initialise le système RAG complet avec les retrievers et la chaîne de traitement.
    Utilise le cache de Streamlit pour éviter de réinitialiser le système à chaque fois.
    """
    print(" Initialisation du système RAG...")
    
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return None, None
    
    # Configuration des retrievers
    print(" Configuration des retrievers...")
    semantic_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 10
        }
    )
    
    # Récupération des documents pour BM25
    try:
        collection = vectorstore._collection
        results = collection.get()
        documents = []
        for i in range(len(results['ids'])):
            doc = Document(
                page_content=results['documents'][i],
                metadata=results['metadatas'][i] if results['metadatas'] else {}
            )
            documents.append(doc)
        
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 10
    except Exception as e:
        print(f"Erreur lors de la création du BM25Retriever : {str(e)}")
        bm25_retriever = None

    if bm25_retriever is None:
        # Si BM25 échoue, on utilise uniquement le retriever sémantique
        hybrid_retriever = semantic_retriever
    else:
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[0.5, 0.5] 
        )
    print(" Retrievers configurés")

    
    llm = get_llm(st.session_state.params)
    if llm is None:
        return None, None

# **Instructions de traitement de la question :**
# 1.  **Synthèse Factuelle :** Extrayez et synthétisez toutes les informations factuelles et pertinentes du CONTEXTE qui répondent directement à la question, en intégrant les éléments clés des articles de loi si approprié.
# 2.  **Gestion de l'Absence d'Information :** Si, après avoir examiné le CONTEXTE, vous ne trouvez pas de réponse pertinente ou suffisante, indiquez clairement : "Je n'ai pas trouvé la réponse pertinente dans les documents fournis."
# 3.  **Objectivité :** Restez toujours objectif. Ne fournissez jamais de conseils juridiques, d'opinions personnelles, d'interprétations subjectives, ou d'informations qui ne proviennent pas du CONTEXTE.

    # Configuration de la chaîne RAG
    print("⚙️ Configuration de la chaîne RAG...")
    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""
Vous êtes un assistant juridique expert. Votre mission est de fournir des réponses précises et complètes basées sur le CONTEXTE fourni.

**Instructions :**
1. Analysez le CONTEXTE pour trouver toutes les informations pertinentes
2. Fournissez une réponse détaillée qui couvre tous les aspects de la question
3. Incluez les références aux articles pertinents
4. Si certaines informations sont manquantes, indiquez-le clairement

**Format de la Réponse :**
- Réponse principale
- Détails supplémentaires si nécessaire
- Références aux articles
- Limitations ou points à noter
        """),
        HumanMessagePromptTemplate.from_template("""
CONTEXTE:
{context}

QUESTION:
{question}

RÉPONSE PRÉCISE ET BASÉE SUR LE CONTEXTE:
        """),
    ])

    multi_query_template = """Vous êtes un assistant juridique spécialisé dans le Code du Travail français.
Votre tâche est de générer des requêtes de recherche pertinentes basées sur la question de l'utilisateur.

Générez 4-5 requêtes de recherche qui permettront de trouver les articles et sections pertinents du Code du Travail.
Les requêtes peuvent :
- Être plus générales ou plus spécifiques que la question originale
- Inclure des termes juridiques clés
- Explorer différents aspects de la question
- Utiliser des synonymes ou des formulations alternatives

Question originale : {question}

Requêtes de recherche :"""

    multi_query_prompt = ChatPromptTemplate.from_template(multi_query_template)

    # multi_query_llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash-preview-05-20",
    #     temperature=0.4, 
    #     max_output_tokens=1024,
    #     top_p=0.8
    # )

    multi_query_chain = multi_query_prompt | llm| StrOutputParser() | (lambda x: x.split("\n"))

    prepare_inputs = {
        "question_originale": RunnablePassthrough(),
        "questions_pour_retrieval": multi_query_chain,
    }

    retrieval_and_aggregation_chain = (
        RunnableLambda(lambda x: x["questions_pour_retrieval"])
        | RunnableLambda(lambda queries: [hybrid_retriever.invoke(q) for q in queries])
        | RunnableLambda(lambda lists_of_docs: [doc for sublist in lists_of_docs for doc in sublist])
        | RunnableLambda(
            lambda docs: list(
                {
                    (doc.page_content, frozenset(doc.metadata.items())): doc
                    for doc in docs
                }.values()
            )
        )
    )

    context_and_question_for_llm = {
        "context": retrieval_and_aggregation_chain,
        "question": RunnableLambda(lambda x: x["question_originale"])
    }

    generation_chain = qa_prompt | llm | StrOutputParser()

    rag_chain_with_sources = (
        prepare_inputs
        | RunnableParallel(
            response=context_and_question_for_llm | generation_chain,
            source_documents=context_and_question_for_llm | RunnableLambda(lambda x: x["context"])
        )
    ).with_config(run_name="RAG Chain with Multi-Query & Self-Query")
    print(" Chaîne RAG configurée")

    print(" Initialisation du système RAG terminée avec succès!")
    return rag_chain_with_sources, hybrid_retriever

# --- Interface de Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialisation du système RAG
with st.spinner("Initialisation du système RAG..."):
    rag_chain, retriever = initialize_rag_system()

if rag_chain is None:
    st.error("Le système RAG n'a pas pu être initialisé correctement.")
    st.info("Veuillez vérifier les logs pour plus de détails.")
    st.stop()

st.success(" Assistant juridique prêt à l'emploi !")

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Interface de saisie
if prompt_input := st.chat_input("Posez votre question sur le Code du Travail..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("Recherche de réponse..."):
            try:
                full_response = rag_chain.invoke(prompt_input)
                st.markdown(full_response["response"])
                st.session_state.messages.append({"role": "assistant", "content": full_response["response"]})
            except Exception as e:
                st.error(f"Erreur lors de la génération de la réponse : {str(e)}")
                st.info("Veuillez réessayer ou reformuler votre question.")

# Sidebar pour les paramètres
with st.sidebar:
    st.markdown("""
        # Devient le Harvey Specter de demain 😎​

        ###📱​ Version 1.0.0

        """)
    st.divider()

    if 'show_authors' not in st.session_state:
        st.session_state.show_authors = False
        st.session_state.authors_start_time = None

    if st.button("Autors"):
        st.session_state.show_authors = True
        st.session_state.authors_start_time = time.time()

    if st.session_state.show_authors:
        current_time = time.time()
        if current_time - st.session_state.authors_start_time < 60:
            st.markdown("""
             - Jefferson 
             - Ruben
             - Lucas
             - Haiayan
             - Henoc
            """)
        else:
            st.session_state.show_authors = False

    st.divider()

    st.header("Options")
    st.subheader("Paramètres du modèle")

    model = st.selectbox(
        "Modèle Gemini",
        options=["gemini-2.0-flash", "gemini-1.5-pro"],
        index=0
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.params['temperature'],
        step=0.1
    )

    top_p = st.slider(
        "Top_p",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.params['top_p'],
        step=0.1
    )

    # Mise à jour des paramètres
    if (model != st.session_state.params['model'] or 
        temperature != st.session_state.params['temperature'] or 
        top_p != st.session_state.params['top_p']):
        
        st.session_state.params['model'] = model
        st.session_state.params['temperature'] = temperature
        st.session_state.params['top_p'] = top_p
        st.rerun()

    if st.button("Effacer l'historique du chat"):
        st.session_state.messages = []
        st.rerun() 
    
    st.divider()
    
    st.markdown("""

    ### 📚 À propos
    Cette application utilise un système RAG (Retrieval-Augmented Generation) pour répondre aux questions sur le Code du Travail.
    Pour cette application, nous avons utilisé :
    - Streamlit pour l'interface
    - Gemini comme LLM
    - ChromaDB comme base de données vectorielles
    - Langchain comme orchestrateur
    - Ragas pour l'évaluation avec les métriques : Context Relevancy, Faithfulness, AnswerAccuracy
    
    ### 🔍 Fonctionnalités 
    - Recherche sémantique dans le Code du Travail
    - Génération de réponses précises et contextuelles
    - Personnalisation des paramètres du modèle
    
    ### ⚠️ Limitations
    - Les réponses sont basées uniquement sur le Code du Travail fourni
    - Ne remplace pas un conseil juridique professionnel
    - Les réponses peuvent varier selon les paramètres choisis
    """) 

