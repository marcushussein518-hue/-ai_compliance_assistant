# Install Homebrew in MAcOS Terminal
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#
# Install pyev in MacOS terminal
# brew update
# brew install pyenv
#
# Then add pyenv to your shell (zsh):
# echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
# echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
# echo 'eval "$(pyenv init -)"' >> ~/.zshrc
# source ~/.zshrc
#
# Install Python 3.11 (the one recommended for LangChain/FastAPI)
# pyenv install 3.11.7
#
# Set Python 3.11 as the default for your project
# cd ai_compliance_assistant
#
# Set local python version:
# pyenv local 3.11.7
# 
# python --version
# Python 3.11.7
#
# If your Python 3.14 was installed manually (e.g., from python.org),remove it safely:
# sudo rm -rf /Library/Frameworks/Python.framework/Versions/3.14
# sudo rm -rf "/Applications/Python 3.14"
# Then remove symlinks:
# sudo rm /usr/local/bin/python3.14
# sudo rm /usr/local/bin/pip3.14
# 
# Check python version
# python3 --version
#
# Re-install dependencies
# pip install langchain openai chromadb fastapi 
# uvicorn pypdf python-dotenv
# 
# pip3 show langchain
# Name: langchain
# Version: 1.0.8
# 
# Missing Installation: The langchain_text_splitters package,
# install the package using pip:
# pip install langchain-text-splitters
#
# In modern LangChain versions (0.1–0.3+), 
# the module was moved to: 
# langchain_text_splitters   (plural)

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)

#
# See debug.chunks.py
#
# STEP 3 — Create Embeddings (OpenAI)
#
# Use: 
# from langchain_openai import OpenAIEmbeddings
#
# Install the langchain-openai package
# pip install langchain-openai
#

from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# STEP 4 — Save to Vector Store (ChromaDB)
#
# NOT: 
# from langchain.vectorstores import Chroma   
#
# LangChain library was refactored, splitting core functionalities into langchain-core, langchain-community, and langchain with the stable 0.1 release.
# pip install langchain-community

from langchain_community.vectorstores import Chroma

def build_vectorstore(chunks):
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="compliance_docs"
    )
    return vectorstore

# STEP 5 — Querying (answer_question)
#
# NOT: 
# from langchain.llms import OpenAI   # breaks in v1.0+

from langchain_openai import OpenAI

llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def answer_question(query, vectorstore):
    # Step 1: Retrieve relevant chunks
    docs = vectorstore.similarity_search(query, k=3)

    # Step 2: Combine retrieved text
    context = "\n\n".join([d.page_content for d in docs])

    # Step 3: Ask LLM to answer based on retrieved context
    prompt = f"""
    You are a compliance assistant. Answer ONLY using the information below:

    {context}

    Question: {query}
    """

    response = llm(prompt)
    return response


