{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "id": "px5zDs9YItjJ"
   },
   "outputs": [],
   "source": [
    "#reference Hands-On Applications with LangChain, Pinecone, and OpenAI. Build Web Apps with Streamlit. Join the AI Revolution Today! https://drive.google.com/drive/folders/1Ein0oHa-eAyLNC-dat73G6OC6SnPX6lJ"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "PdvEKB6qo2J6"
   },
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from langchain.embeddings.openai import OpenAIEmbeddings\n",
    "from langchain.vectorstores import Chroma"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "id": "x8jjlzqCo5nK"
   },
   "outputs": [],
   "source": [
    "# loading PDF, DOCX and TXT files as LangChain Documents\n",
    "def load_document(file):\n",
    "    import os\n",
    "    name, extension = os.path.splitext(file)\n",
    "\n",
    "    if extension == '.pdf':\n",
    "        from langchain.document_loaders import PyPDFLoader\n",
    "        print(f'Loading {file}')\n",
    "        loader = PyPDFLoader(file)\n",
    "    elif extension == '.docx':\n",
    "        from langchain.document_loaders import Docx2txtLoader\n",
    "        print(f'Loading {file}')\n",
    "        loader = Docx2txtLoader(file)\n",
    "    elif extension == '.txt':\n",
    "        from langchain.document_loaders import TextLoader\n",
    "        loader = TextLoader(file)\n",
    "    else:\n",
    "        print('Document format is not supported!')\n",
    "        return None\n",
    "\n",
    "    data = loader.load()\n",
    "    return data\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "id": "7nPvzCcQo7Mg"
   },
   "outputs": [],
   "source": [
    "# splitting data in chunks\n",
    "def chunk_data(data, chunk_size=256, chunk_overlap=20):\n",
    "    from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
    "    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)\n",
    "    chunks = text_splitter.split_documents(data)\n",
    "    return chunks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "id": "Gsnz_qVTgO9z"
   },
   "outputs": [],
   "source": [
    "# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store\n",
    "def create_embeddings(chunks):\n",
    "    embeddings = OpenAIEmbeddings()\n",
    "    vector_store = Chroma.from_documents(chunks, embeddings)\n",
    "\n",
    "    # if you want to use a specific directory for chromadb\n",
    "    # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')\n",
    "    return vector_store"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "id": "oBYJuL4TgRcM"
   },
   "outputs": [],
   "source": [
    "#larger K size would provide more elborate answers but cost more\n",
    "\n",
    "\n",
    "def ask_and_get_answer(vector_store, q, k=3):\n",
    "    from langchain.chains import RetrievalQA\n",
    "    from langchain.chat_models import ChatOpenAI\n",
    "\n",
    "    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)\n",
    "    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})\n",
    "    chain = RetrievalQA.from_chain_type(llm=llm, chain_type=\"stuff\", retriever=retriever)\n",
    "\n",
    "    answer = chain.run(q)\n",
    "    return answer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "id": "6a30py7bgXTv"
   },
   "outputs": [],
   "source": [
    "# calculate embedding cost using tiktoken\n",
    "def calculate_embedding_cost(texts):\n",
    "    import tiktoken\n",
    "    enc = tiktoken.encoding_for_model('text-embedding-ada-002')\n",
    "    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])\n",
    "    # print(f'Total Tokens: {total_tokens}')\n",
    "    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')\n",
    "    return total_tokens, total_tokens / 1000 * 0.0004\n",
    "\n",
    "\n",
    "# clear the chat history from streamlit session state\n",
    "def clear_history():\n",
    "    if 'history' in st.session_state:\n",
    "        del st.session_state['history']\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "fOf0v17Jgf3h",
    "outputId": "46eecd0a-ad9e-4b6e-aac3-dde297bb8772",
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-10-24 09:03:10.228 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\chili\\Desktop\\Emma\\python\\streamlist\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "OpenAI API Key: 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\n"
     ]
    }
   ],
   "source": [
    " # loading the OpenAI api key from .env\n",
    "from dotenv import load_dotenv, find_dotenv\n",
    "load_dotenv(find_dotenv(), override=True)\n",
    "\n",
    "st.image('img.png')\n",
    "st.subheader('LLM Question-Answering Application 🤖')\n",
    "with st.sidebar:\n",
    "# text_input for the OpenAI API key (alternative to python-dotenv and .env)\n",
    "\n",
    "\n",
    "\n",
    "# Define the path to the .env file\n",
    "  #   dotenv_path = '/content/gdrive/MyDrive/data/env_vars.env'\n",
    "\n",
    "\n",
    "# Load environment variables from .env file\n",
    "    #load_dotenv(dotenv_path)\n",
    "#load_dotenv('dotenv_path')\n",
    "\n",
    "# Access the environment variables\n",
    "\n",
    "#api_key=os.environ.get(\"OPENAI_API_KEY\")\n",
    "#print(api_key)\n",
    "     api_key = os.getenv(\"OPENAI_API_KEY\")\n",
    "     print(\"OpenAI API Key:\", api_key)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "id": "GF6vdGjNjwqg"
   },
   "outputs": [],
   "source": [
    " # file uploader widget\n",
    "uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])\n",
    "\n",
    " # chunk size number widget\n",
    "chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)\n",
    "\n",
    "# k number input widget\n",
    "k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)\n",
    "\n",
    " # add data button widget\n",
    "add_data = st.button('Add Data', on_click=clear_history)\n",
    "if uploaded_file and add_data: # if the user browsed a file\n",
    "     with st.spinner('Reading, chunking and embedding file ...'):\n",
    "\n",
    "# writing the file from RAM to the current directory on disk\n",
    "      bytes_data = uploaded_file.read()\n",
    "      file_name = os.path.join('./', uploaded_file.name)\n",
    "      with open(file_name, 'wb') as f:\n",
    "           f.write(bytes_data)\n",
    "\n",
    "      data = load_document(file_name)\n",
    "      chunks = chunk_data(data, chunk_size=chunk_size)\n",
    "      st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')\n",
    "\n",
    "      tokens, embedding_cost = calculate_embedding_cost(chunks)\n",
    "      st.write(f'Embedding cost: ${embedding_cost:.4f}')\n",
    "\n",
    "   # creating the embeddings and returning the Chroma vector store\n",
    "      vector_store = create_embeddings(chunks)\n",
    "\n",
    "      # saving the vector store in the streamlit session state (to be persistent between reruns)\n",
    "      st.session_state.vs = vector_store\n",
    "      st.success('File uploaded, chunked and embedded successfully.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "id": "xzrK3gqvkd5K"
   },
   "outputs": [],
   "source": [
    " # user's question text input widget\n",
    "q = st.text_input('Ask a question about the content of your file:')\n",
    "if q: # if the user entered a question and hit enter\n",
    "   if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)\n",
    "    vector_store = st.session_state.vs\n",
    "    st.write(f'k: {k}')\n",
    "    answer = ask_and_get_answer(vector_store, q, k)\n",
    "\n",
    "            # text area widget for the LLM answer\n",
    "    st.text_area('LLM Answer: ', value=answer)\n",
    "\n",
    "    st.divider()\n",
    "\n",
    "            # if there's no chat history in the session state, create it\n",
    "    if 'history' not in st.session_state:\n",
    "               st.session_state.history = ''\n",
    "\n",
    "            # the current question and answer\n",
    "               value = f'Q: {q} \\nA: {answer}'\n",
    "\n",
    "               st.session_state.history = f'{value} \\n {\"-\" * 100} \\n {st.session_state.history}'\n",
    "    h = st.session_state.history\n",
    "\n",
    "            # text area widget for the chat history\n",
    "    st.text_area(label='Chat History', value=h, key='history', height=400)\n",
    "\n",
    "# run the app: streamlit run ./chat_with_documents.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
