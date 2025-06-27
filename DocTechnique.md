# 📚 Documentation Technique – Assistant Juridique RAG (Code du Travail)

## 1. Présentation du Projet

Ce projet implémente un assistant juridique basé sur le paradigme **RAG** (Retrieval-Augmented Generation) pour répondre à des questions sur le Code du Travail français. Il combine une **recherche hybride** (sémantique + lexicale) et la **génération de réponses contextualisées** via un **LLM (Gemini)**, le tout orchestré avec **LangChain** et une interface **Streamlit**.

---

## 2. Architecture du Système

- **Chargement & Découpage** : Extraction du texte du PDF, découpage intelligent en *chunks* enrichis de métadonnées (page, titre, article, etc.).
- **Vectorisation & Indexation** : Génération d’**embeddings** pour chaque chunk et stockage dans **ChromaDB** (base vectorielle persistante).
- **Recherche Hybride** : Combinaison d’un **retriever sémantique** (embeddings) et d’un **retriever BM25** (lexical) pour maximiser la pertinence.
- **Génération** : Utilisation de **Gemini** (via LangChain) pour synthétiser une réponse à partir des chunks récupérés.
- **Interface Utilisateur** : Application web interactive avec **Streamlit**.

---

## 3. Fonctionnement Global

### 🔹 Chargement du PDF
Le fichier PDF est chargé depuis `data` et découpé en chunks à l’aide de règles adaptées à la structure du Code du Travail (articles, titres, chapitres…).

### 🔹 Vectorisation & Indexation
Chaque chunk est transformé en **embedding** (modèle HuggingFace multilingue) et stocké dans **ChromaDB** avec ses métadonnées.

### 🔹 Recherche Hybride
Lorsqu’une question est posée :
- Recherche sémantique (similarité vectorielle)
- Recherche lexicale (BM25)
- Fusion des résultats pour une meilleure couverture

### 🔹 Génération de la Réponse
Les chunks les plus pertinents sont transmis à **Gemini** via **LangChain**, qui génère une réponse synthétique et cite les sources.

### 🔹 Affichage & Interaction
L’utilisateur interagit via **Streamlit**, peut ajuster les paramètres du modèle et visualiser les sources des réponses.

---

## 4. Principales Fonctions et Modules

- a. Chargement et Découpage 
```
@st.cache_data(show_spinner=False)
def load_documents() -> Optional[List[Any]]:
    """
    Charge le PDF et retourne une liste de documents (pages).
    Utilise PyPDFLoader.
    """
```

```
@st.cache_data(show_spinner=False)
def load_documents() -> Optional[List[Any]]:
    """
    Charge le PDF et retourne une liste de documents (pages).
    Utilise PyPDFLoader.
    """
```
- b. Embeddings et Base Vectorielle
```
@st.cache_resource
def get_embedding_model():
    """
    Initialise le modèle d'embeddings HuggingFace.
    """
``` 

```
@st.cache_resource
def get_vectorstore():
    """
    Crée ou charge la base ChromaDB persistante.
    """
```
- c. Génération et Chaîne RAG 
```
@st.cache_resource
def get_llm(_params: Dict[str, Any]) -> Optional[Any]:
    """
    Initialise le modèle Gemini (via LangChain).
    """
```

```
@st.cache_resource
def initialize_rag_system():
    """
    Configure la chaîne RAG complète (retrievers hybrides, prompt, LLM).
    """
```

---

## 5. Paramétrage et Interface

- **Modèle Gemini** : Sélection entre plusieurs versions (ex. `gemini-2.0-flash`, `gemini-1.5-pro`)
- **Temperature** : Contrôle la créativité du LLM (0 = factuel, 1 = créatif)
- **Top_p** : Diversité des réponses
- **Historique de chat** : Conservation des échanges utilisateur/assistant

---

## 6. Gestion des Erreurs & Robustesse

- Validation des entrées (PDF, clés API)
- Gestion des erreurs de chargement (PDF, embeddings, ChromaDB)
- Fallback automatique sur le retriever sémantique si BM25 échoue
- Messages d’erreur utilisateur clairs dans l’interface
- Mise en cache pour accélérer les traitements et éviter les rechargements inutiles

---

## 7. Optimisations

- **Performance** : Caching Streamlit, embeddings multilingues optimisés, découpage intelligent
- **Qualité des réponses** : Fusion des retrievers, prompt engineering spécialisé, gestion des sources
- **Scalabilité** : ChromaDB persistante, découpage modulaire du code

---

## 8. Évaluation

- **Ragas** : Évaluation automatique sur trois axes :
  - *Context Relevancy*
  - *Faithfulness*
  - *Answer Accuracy*
- **Jeu de données d’évaluation** : Fichier `eval_dataset.json` avec questions et réponses attendues

---

## 9. Déploiement

### 🔹 Lancement local :

**Prérequis** :
- Python 3.8+
- Fichier `.env` avec les clés API Google et LangChain
- PDF du Code du Travail dans `data`

**Installation des dépendances** :
```bash
pip install -r requirements.txt
```
---

## 10. Structure des Dossiers
```bash
data/                    # PDF(s) et jeu d’évaluation
chroma_db_codetravail/   # Base vectorielle persistante
rag_system.py            # Script principal Streamlit
.env                     # Clés API
requirements.txt         # Dépendances
```

---

## 11. Technologies Utilisées 🛠️
| Technologie     | Rôle                                                         |
| --------------- | ------------------------------------------------------------ |
| **Streamlit**   | Interface utilisateur interactive                            |
| **LangChain**   | Orchestration de la chaîne RAG (Recherche + Génération)      |
| **ChromaDB**    | Base de données vectorielle persistante                      |
| **Gemini**      | LLM (Large Language Model) utilisé pour générer les réponses |
| **HuggingFace** | Génération des embeddings multilingues                       |
| **Ragas**       | Évaluation automatique des réponses générées                 |



---

## 12. Auteurs 👥
- Jefferson
- Ruben
- Lucas
- Henoc
- Haiayan

