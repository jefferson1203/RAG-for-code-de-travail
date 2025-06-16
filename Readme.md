# Assistant Juridique RAG - Code du Travail

[Asistant juridique 🚀](https://bob-lo21p25.streamlit.app/)

Un assistant juridique intelligent basé sur un système RAG (Retrieval-Augmented Generation) pour répondre aux questions sur le Code du Travail français.

## 🚀 Fonctionnalités

- Recherche sémantique dans le Code du Travail
- Génération de réponses précises et contextuelles
- Interface utilisateur intuitive avec Streamlit
- Personnalisation des paramètres du modèle
- Système hybride de recherche (BM25 + Sémantique)

## 🛠️ Technologies Utilisées

- **Streamlit** : Interface utilisateur
- **Gemini** : Modèle de langage (LLM)
- **ChromaDB** : Base de données vectorielle
- **LangChain** : Orchestration du système RAG
- **Ragas** : Évaluation des performances

## 📋 Prérequis

1. Python 3.8+
2. Clés API requises :
   - Google API Key (pour Gemini)
   - LangChain API Key (pour le tracing)

## 🔧 Installation

1. Clonez le repository :
```bash
git clone [URL_DU_REPO]
cd [NOM_DU_DOSSIER]
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Créez un fichier `.env` à la racine du projet avec vos clés API :
```env
GOOGLE_API_KEY=votre_clé_google
LANGCHAIN_API_KEY=votre_clé_langchain
```

4. Placez votre fichier PDF du Code du Travail dans le dossier `data/` :
```bash
mkdir data
# Copiez votre fichier PDF dans le dossier data/
```

## 🚀 Lancement

Pour démarrer l'application :
```bash
streamlit run rag_system.py
```

## ⚙️ Configuration

### Paramètres du Modèle

Dans l'interface, vous pouvez ajuster :
- **Modèle Gemini** : Choisir entre gemini-2.0-flash et gemini-1.5-pro
- **Temperature** : Contrôle la créativité des réponses (0 = plus précis, 1 = plus créatif)
- **Top_p** : Contrôle la diversité des réponses

### Structure des Données

- `data/` : Contient le PDF du Code du Travail
- `chroma_db_codetravail/` : Base de données vectorielle persistante

## 🔍 Fonctionnement

1. **Chargement des Documents** :
   - Le PDF est chargé et divisé en chunks
   - Les chunks sont vectorisés et stockés dans ChromaDB

2. **Système de Recherche** :
   - Recherche sémantique via embeddings
   - Recherche lexicale via BM25
   - Combinaison des résultats pour une meilleure pertinence

3. **Génération de Réponses** :
   - Le contexte pertinent est récupéré
   - Gemini génère une réponse basée sur le contexte
   - Les références aux articles sont incluses

## 🧪 Évaluation

Le système est évalué avec Ragas sur trois métriques :
- Context Relevancy
- Faithfulness
- Answer Accuracy

## ⚠️ Limitations

- Les réponses sont basées uniquement sur le Code du Travail fourni
- Ne remplace pas un conseil juridique professionnel
- Les réponses peuvent varier selon les paramètres choisis

## 🔧 Dépannage

### Problèmes Courants

1. **Erreur HNSW Index** :
   - Supprimez le dossier `chroma_db_codetravail/`
   - Relancez l'application pour recréer la base

2. **Erreur de Chargement du PDF** :
   - Vérifiez que le fichier existe dans `data/`
   - Assurez-vous que le PDF n'est pas corrompu

3. **Erreur API** :
   - Vérifiez vos clés API dans le fichier `.env`
   - Assurez-vous que les clés sont valides

## 👥 Auteurs

- Jefferson
- Ruben
- Lucas
- Haiayan
- Henoc


## 📚 Documentation Technique

### Fonctions Principales

#### 1. Chargement et Préparation des Documents
```python
@st.cache_data(show_spinner=False)
def load_documents() -> Optional[List[Any]]:
    """
    Charge le document PDF et retourne une liste de documents.
    - Utilise PyPDFLoader pour charger le PDF
    - Met en cache les résultats pour éviter de recharger à chaque fois
    - Retourne None en cas d'erreur
    """
```

```python
@st.cache_data(show_spinner=False)
def split_documents(_documents: List[Any]) -> Optional[List[Any]]:
    """
    Divise les documents en chunks plus petits pour le traitement.
    - Utilise RecursiveCharacterTextSplitter avec des patterns spécifiques au Code du Travail
    - Patterns de découpage : Articles, Sections, Chapitres, Titres
    - Taille des chunks : 800 caractères avec 100 caractères de chevauchement
    """
```

#### 2. Gestion des Embeddings et Base Vectorielle
```python
@st.cache_resource
def get_embedding_model():
    """
    Initialise le modèle d'embeddings.
    - Utilise HuggingFaceEmbeddings avec le modèle 'paraphrase-multilingual-MiniLM-L12-v2'
    - Optimisé pour le français et le multilinguisme
    - Normalise les embeddings pour une meilleure performance
    """
```

```python
@st.cache_resource
def get_vectorstore():
    """
    Gère la base de données vectorielle ChromaDB.
    - Crée ou charge une base existante
    - Gère la persistance des données
    - Implémente la récupération des documents
    """
```

#### 3. Configuration du LLM et Chaîne RAG
```python
@st.cache_resource
def get_llm(_params: Dict[str, Any]) -> Optional[Any]:
    """
    Initialise le modèle de langage Gemini.
    - Configure les paramètres (temperature, top_p)
    - Gère les tokens de sortie
    - Optimise les performances
    """
```

```python
@st.cache_resource
def initialize_rag_system():
    """
    Initialise le système RAG complet.
    - Configure les retrievers (sémantique + BM25)
    - Met en place la chaîne de traitement
    - Gère le prompt template et la génération
    """
```

### Architecture du Système RAG

1. **Préparation des Données**
   - Chargement du PDF
   - Découpage intelligent
   - Vectorisation des chunks

2. **Système de Recherche**
   - Retriever sémantique (embeddings)
   - Retriever BM25 (recherche lexicale)
   - Combinaison des résultats

3. **Génération de Réponses**
   - Prompt template spécialisé
   - Contexte enrichi
   - Génération avec Gemini

### Prompts et Templates

```python
qa_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
    Vous êtes un assistant juridique expert...
    """),
    HumanMessagePromptTemplate.from_template("""
    CONTEXTE:
    {context}
    QUESTION:
    {question}
    """)
])
```

### Gestion des Erreurs

Le système implémente plusieurs niveaux de gestion d'erreurs :
1. Validation des entrées
2. Gestion des erreurs de chargement
3. Fallback sur le retriever sémantique
4. Messages d'erreur utilisateur

### Optimisations

1. **Performance**
   - Mise en cache des résultats
   - Optimisation des embeddings
   - Gestion efficace de la mémoire

2. **Qualité des Réponses**
   - Découpage intelligent des documents
   - Combinaison de retrievers
   - Prompt engineering spécialisé

3. **Robustesse**
   - Gestion des erreurs
   - Fallback mechanisms
   - Validation des données