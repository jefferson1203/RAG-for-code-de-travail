# Projet RAG - Code du Travail

## Description
Ce projet implémente un système RAG (Retrieval Augmented Generation) dédié au Code du travail, servant d'outil de requêtage intelligent et d'aide à la décision pour des cabinets de consultation juridique.

## Objectifs
- Permettre aux utilisateurs (juristes, consultants, avocats juniors) de poser des questions en langage naturel
- Obtenir des réponses précises, contextualisées et appuyées par des sources juridiques fiables
- Fournir un accès rapide et pertinent aux informations du Code du travail

## Architecture du Projet

### 1. Préparation des Données
- **Chargement du Document** : Utilisation de PyPDFLoader pour charger le Code du travail
- **Découpage Intelligent** : 
  - `TitleBasedSplitter` : Découpage basé sur la structure des titres
  - `CodeDuTravailStructureExtractor` : Extraction de la structure hiérarchique (Partie, Livre, Titre, Chapitre, Section, Article)

### 2. Système RAG
- **Embeddings** : Utilisation de HuggingFaceEmbeddings (modèle "all-MiniLM-L6-v2")
- **Base de Données Vectorielle** : ChromaDB pour le stockage et la recherche des embeddings
- **LLM** : Intégration de Gemini-Pro pour la génération des réponses
- **Chaîne RAG** : Orchestration via LangChain

### 3. Interface Utilisateur
- Interface web interactive via Streamlit
- Affichage des réponses avec leurs sources
- Gestion des erreurs et feedback utilisateur

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd [NOM_DU_REPO]
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurer les variables d'environnement :
```bash
# Créer un fichier .env avec les clés API suivantes
LANGCHAIN_API_KEY=votre_clé_langchain
GOOGLE_API_KEY=votre_clé_google
PDF_PATH=chemin/vers/code_du_travail.pdf
```

## Utilisation

1. Lancer l'application :
```bash
python projet_rag_ruben.py
```

2. Accéder à l'interface web :
- Ouvrir un navigateur
- Aller à l'URL indiquée dans la console (généralement http://localhost:8501)

3. Poser des questions :
- Utiliser le champ de texte pour saisir votre question
- La réponse sera générée avec les sources pertinentes

## Structure du Code

```
projet_rag_ruben.py
├── Configuration
│   ├── Variables d'environnement
│   └── Paramètres globaux
├── Classes Personnalisées
│   ├── TitleBasedSplitter
│   └── CodeDuTravailStructureExtractor
├── Fonctions Utilitaires
│   └── format_docs_with_sources
└── Fonction Principale
    └── main()
```

## Fonctionnalités Clés

1. **Découpage Intelligent**
   - Préservation de la structure hiérarchique
   - Conservation du contexte
   - Gestion des métadonnées

2. **Recherche Sémantique**
   - Embeddings de haute qualité
   - Recherche de similarité vectorielle
   - Récupération des sources pertinentes

3. **Génération de Réponses**
   - Réponses contextualisées
   - Citations des sources
   - Gestion des hallucinations

## Dépendances Principales

```bash
# Core dependencies
langchain>=0.1.0
langchain-community>=0.0.10
langchain-openai>=0.0.2
langchain-huggingface>=0.0.1  # Nouveau package pour HuggingFaceEmbeddings
openai>=1.3.0
chromadb>=0.4.18
sentence-transformers>=2.2.2
python-dotenv>=1.0.0
streamlit>=1.29.0

# Protobuf version spécifique pour éviter les conflits
protobuf==3.20.0  # Version fixe pour éviter les problèmes de compatibilité
```

## Dépannage

### 1. Erreur de Descriptors Protobuf
Si vous rencontrez l'erreur :
```
TypeError: Descriptors cannot be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be
regenerated with protoc >= 3.19.0.
```

Solutions :
1. Installer la version spécifique de protobuf :
```bash
pip install protobuf==3.20.0
```

2. Ou définir la variable d'environnement :
```bash
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### 2. Erreur HuggingFaceEmbeddings
Si vous voyez l'avertissement :
```
LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated
```

Solution :
1. Installer le nouveau package :
```bash
pip install langchain-huggingface
```

2. Mettre à jour l'import dans le code :
```python
from langchain_huggingface import HuggingFaceEmbeddings
```

### 3. Erreur ChromaDB
Si vous rencontrez des problèmes avec ChromaDB :
1. Désinstaller et réinstaller ChromaDB :
```bash
pip uninstall chromadb
pip install chromadb==0.4.18
```

2. Vérifier que le dossier de persistance existe :
```python
import os
os.makedirs("./chroma_db_codetravail", exist_ok=True)
```

### 4. Erreur Streamlit
Si vous rencontrez des problèmes avec Streamlit :
1. Vérifier la version de Python (recommandé : Python 3.8-3.10)
2. Réinstaller Streamlit :
```bash
pip uninstall streamlit
pip install streamlit==1.29.0
```

## Configuration Recommandée

1. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

2. Installer les dépendances dans l'ordre :
```bash
pip install -r requirements.txt
```

3. Vérifier les variables d'environnement :
```bash
# .env
LANGCHAIN_API_KEY=votre_clé_langchain
GOOGLE_API_KEY=votre_clé_google
PDF_PATH=chemin/vers/code_du_travail.pdf
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request