"""
# -*- coding: utf-8 -*-
# Projet RAG - Code du Travail
# Auteur: Ruben
# Description: Système RAG pour le Code du Travail permettant de répondre aux questions juridiques
#              en utilisant LangChain, ChromaDB, et Gemini.
# Environnement: Cursor IDE
# Dernière mise à jour: 2024
"""

# ============================================================================
# IMPORTS ET CONFIGURATION
# ============================================================================

import os
import re
import sys
from typing import List, Optional, Dict, Any

# Configuration de protobuf pour éviter les erreurs
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['PYTHONPATH'] = os.getcwd()  # Ajout du répertoire courant au PYTHONPATH

# LangChain imports
from langchain.text_splitter import TextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma  # Mise à jour de l'import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # Mise à jour de l'import HuggingFaceEmbeddings
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

# ============================================================================
# CONFIGURATION DES VARIABLES D'ENVIRONNEMENT
# ============================================================================

# Configuration des variables d'environnement pour LangChain et Google
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Chemin vers le fichier PDF du Code du Travail
PDF_PATH = os.getenv('PDF_PATH')

# ============================================================================
# CLASSES PERSONNALISÉES
# ============================================================================

class TitleBasedSplitter(TextSplitter):
    """
    Splitter de texte basé sur la détection de titres.
    Garde les métadonnées de la page d'origine pour chaque chunk.
    """
    def __init__(self, pattern: str = r"(Titre\s+(?:[IVXLCDM]+(?:er|ème)?|[A-Za-z\d\u00C0-\u00FF'-]+)\s*:)"):
        super().__init__()
        self.compiled_pattern = re.compile(pattern, re.IGNORECASE)

    def split_documents(self, documents: List[Document], **kwargs) -> List[Document]:
        all_chunks: List[Document] = []
        for doc in documents:
            text = doc.page_content
            metadata = doc.metadata.copy()

            matches = list(self.compiled_pattern.finditer(text))

            if not matches:
                if text.strip():
                    all_chunks.append(Document(page_content=text.strip(), metadata=metadata))
                continue

            if matches[0].start() > 0:
                pre_title_text = text[0:matches[0].start()].strip()
                if pre_title_text:
                    all_chunks.append(Document(page_content=pre_title_text, metadata=metadata))

            for i in range(len(matches)):
                start = matches[i].start()
                end = matches[i+1].start() if i + 1 < len(matches) else len(text)
                chunk_content = text[start:end].strip()

                if chunk_content:
                    chunk_metadata = metadata.copy()
                    all_chunks.append(Document(page_content=chunk_content, metadata=chunk_metadata))

        return all_chunks

    def split_text(self, text: str) -> List[str]:
        chunks = []
        matches = list(self.compiled_pattern.finditer(text))

        if not matches:
            return [text.strip()] if text.strip() else []

        if matches[0].start() > 0:
            pre_title_text = text[0:matches[0].start()].strip()
            if pre_title_text:
                chunks.append(pre_title_text)

        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i+1].start() if i + 1 < len(matches) else len(text)
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append(chunk_content)
        return chunks

class CodeDuTravailStructureExtractor(TextSplitter):
    """
    Splitter et extracteur de structure pour le Code du Travail.
    Découpe le document en chunks et enrichit les métadonnées avec les informations
    de Partie, Livre, Titre, Chapitre, Section et Article.
    """
    def __init__(
        self,
        title_pattern: str = r"(Titre\s+(?:[IVXLCDM]+(?:er|ème)?|[A-Za-z\d\u00C0-\u00FF'-]+)\s*:\s*.+?)(?=\n(?:Titre\s|Chapitre\s|Section\s|Article\s|$))",
        chapter_pattern: str = r"(Chapitre\s+(?:[IVXLCDM]+(?:er|ème)?|unique|[A-Za-z\d\u00C0-\u00FF'-]+)\s*:\s*.+?)(?=\n(?:Titre\s|Chapitre\s|Section\s|Article\s|$))",
        section_pattern: str = r"(Section\s+(?:\d+|unique|[A-Za-z\d\u00C0-\u00FF'-]+)\s*:\s*.+?)(?=\n(?:Titre\s|Chapitre\s|Section\s|Article\s|$))",
        article_pattern: str = r"(Article\s+((?:L|R|D)\.\s*\d{3}-\d+(?:-\d+)?(?:-\d+)?)[\s\S]*?(?=Article\s+((?:L|R|D)\.\s*\d{3}-\d+(?:-\d+)?(?:-\d+)?)|Titre\s+|Chapitre\s+|Section\s+|$))",
        article_num_capture_group: int = 2,
        keep_separator: bool = True,
        **kwargs
    ):
        super().__init__(keep_separator=keep_separator, **kwargs)
        self.title_pattern = re.compile(title_pattern, re.IGNORECASE | re.DOTALL)
        self.chapter_pattern = re.compile(chapter_pattern, re.IGNORECASE | re.DOTALL)
        self.section_pattern = re.compile(section_pattern, re.IGNORECASE | re.DOTALL)
        self.article_pattern = re.compile(article_pattern, re.IGNORECASE | re.DOTALL)
        self.article_num_capture_group = article_num_capture_group

    def split_documents(self, documents: List[Document]) -> List[Document]:
        all_chunks: List[Document] = []
        current_book = None
        current_part = None
        current_title = None
        current_chapter = None
        current_section = None

        for doc in documents:
            text = doc.page_content
            page_metadata = doc.metadata.copy()

            article_matches = list(self.article_pattern.finditer(text))

            last_idx = 0
            if article_matches and article_matches[0].start() > 0:
                pre_article_text = text[0:article_matches[0].start()].strip()
                if pre_article_text:
                    chunk_metadata = self._extract_hierarchy_metadata(pre_article_text, page_metadata)
                    all_chunks.append(Document(page_content=pre_article_text, metadata=chunk_metadata))
                last_idx = article_matches[0].start()

            for i, match in enumerate(article_matches):
                article_content = match.group(0).strip()
                article_number = match.group(self.article_num_capture_group)

                chunk_metadata = page_metadata.copy()
                chunk_metadata["type"] = "Article"
                chunk_metadata["article_number"] = article_number
                chunk_metadata.update(self._extract_hierarchy_metadata(article_content, page_metadata))

                all_chunks.append(Document(page_content=article_content, metadata=chunk_metadata))
                last_idx = match.end()

            if last_idx < len(text):
                remaining_text = text[last_idx:].strip()
                if remaining_text:
                    chunk_metadata = self._extract_hierarchy_metadata(remaining_text, page_metadata)
                    all_chunks.append(Document(page_content=remaining_text, metadata=chunk_metadata))

        return all_chunks

    def _extract_hierarchy_metadata(self, text: str, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        metadata = base_metadata.copy()

        title_match = self.title_pattern.search(text)
        if title_match:
            metadata["title"] = title_match.group(1).strip()

        chapter_match = self.chapter_pattern.search(text)
        if chapter_match:
            metadata["chapter"] = chapter_match.group(1).strip()

        section_match = self.section_pattern.search(text)
        if section_match:
            metadata["section"] = section_match.group(1).strip()

        return metadata

    def split_text(self, text: str) -> List[str]:
        chunks = []
        article_splits = self.article_pattern.split(text)

        if len(article_splits) > 0 and not self.article_pattern.match(article_splits[0]):
            if article_splits[0].strip():
                chunks.append(article_splits[0].strip())
            start_index = 1
        else:
            start_index = 0

        for i in range(start_index, len(article_splits), self.article_num_capture_group + 1):
            if i + self.article_num_capture_group < len(article_splits):
                article_num = article_splits[i + self.article_num_capture_group -1]
                article_content = article_splits[i + self.article_num_capture_group]

                full_article_chunk = f"Article {article_num.strip()} {article_content.strip()}"
                if full_article_chunk.strip():
                    chunks.append(full_article_chunk.strip())
            elif article_splits[i].strip():
                chunks.append(article_splits[i].strip())
        return chunks

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def ensure_directory_exists(directory: str) -> None:
    """
    S'assure que le répertoire existe, le crée si nécessaire.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Répertoire créé : {directory}")

def format_docs_with_sources(docs: List[Document]) -> str:
    """
    Formate les documents avec leurs sources pour l'affichage.
    """
    formatted_content = ""
    unique_pages = set()

    for i, doc in enumerate(docs):
        formatted_content += f"Contenu source {i+1}:\n{doc.page_content}\n\n"
        if 'page' in doc.metadata:
            page_number = doc.metadata['page'] + 1
            unique_pages.add(str(page_number))

    if unique_pages:
        formatted_content += f"Sources des pages: {', '.join(sorted(list(unique_pages)))}\n"

    return formatted_content

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale qui orchestre le processus RAG.
    """
    try:
        # Vérification du fichier PDF
        if not os.path.exists(PDF_PATH):
            print(f"ERREUR: Le fichier PDF '{PDF_PATH}' n'a pas été trouvé.")
            return

        print(f"Chargement du PDF depuis : {PDF_PATH}")
        
        # Chargement du PDF
        loader = PyPDFLoader(PDF_PATH)
        docs_from_loader = loader.load()

        if not docs_from_loader:
            print("Aucun document n'a pu être chargé.")
            return

        print(f"Nombre de pages chargées : {len(docs_from_loader)}")

        # Extraction de la structure
        structure_extractor = CodeDuTravailStructureExtractor()
        documents = structure_extractor.split_documents(docs_from_loader)
        print(f"Nombre de documents après extraction : {len(documents)}")

        # Configuration des embeddings
        print("Configuration du modèle d'embeddings...")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Configuration de ChromaDB
        print("Configuration de ChromaDB...")
        chroma_db_dir = "./chroma_db_codetravail"
        ensure_directory_exists(chroma_db_dir)
        
        try:
            vectorstore = Chroma.from_documents(
                documents, 
                embedding=embedding_model, 
                persist_directory=chroma_db_dir
            )
            print("Base de données vectorielle créée avec succès.")
        except Exception as e:
            print(f"Erreur lors de la création de la base vectorielle : {e}")
            print("Tentative de récupération de la base existante...")
            vectorstore = Chroma(
                persist_directory=chroma_db_dir,
                embedding_function=embedding_model
            )
        
        # Configuration du retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Configuration du LLM
        print("Configuration du modèle de langage...")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.2)
        
        # Configuration de la chaîne RAG
        print("Configuration de la chaîne RAG...")
        prompt = hub.pull("rlm/rag-prompt")
        retrieval_chain = {"context": retriever, "question": RunnablePassthrough()}
        generation_chain = prompt | llm | StrOutputParser()
        
        rag_chain_with_sources = RunnableParallel(
            response=retrieval_chain | generation_chain,
            source_documents=retrieval_chain | RunnableLambda(lambda x: x["context"])
        ).with_config(run_name="RAG Chain with Sources")

        # Test de la chaîne RAG
        question = "Quelles sont les fonctions du code du travail? Et quelles sont les conditions de représentativité des organisations syndicales ?"
        print(f"\nQuestion : {question}")

        result = rag_chain_with_sources.invoke(question)
        print("\nRéponse Générée par Gemini :")
        print(result["response"])
        print("\nSources :")
        print(format_docs_with_sources(result["source_documents"]))

    except Exception as e:
        print(f"ERREUR lors de l'exécution : {e}")
        print("Traceback complet :")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()