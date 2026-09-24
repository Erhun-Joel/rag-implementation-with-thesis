# ---------------------------- Importing neccessary modules ------------------------------------------
# Importing pdf related functions
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader

# Loading API Interaction libraries
from dotenv import load_dotenv

# Importing functions related to retriever
from langchain_core.tools import retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Importing LLM interaction modules
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Other modules
from os import environ
from langchain_core.runnables import RunnablePassthrough
from rich.markdown import Markdown

# ---------------------------- Working up the document vector space ----------------------------------
# 
# Reading pdf file into Langchain document
body_document = PyPDFLoader('Data/Final Project Body.pdf').load()

# Loading API keys here
load_dotenv('.env')

# Creating document splits
document_chunks = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 300
).split_documents(body_document)

# Create embedding instructions
embeddings = GoogleGenerativeAIEmbeddings(
    model = 'gemini-embedding-2',
    api_key = environ['GEMINI_API_KEY']
)

# Creating an vector space with chroma and gemini
vector_space = Chroma.from_documents(
    documents = document_chunks,
    embedding = embeddings
)

# Creating a retriving function to output context in strings
def retrieving_function(y):
    z = vector_space.as_retriever(search_kwargs = {'k' : 6}).invoke(y)

    x = '\n'

    for i in z:
        x = x + i.page_content + '\n'
    return x

# ---------------------------- Working up LLM querying functions -------------------------------------
# Using gemini-3.6-flash free tier model
model_querying = ChatGoogleGenerativeAI(
    model = 'gemini-3.6-flash',
    temperature = 0,
    api_key = environ['GEMINI_API_KEY']
)

# ---------------------------- Putting everything together to form RAG Chain -------------------------
# Declaring prompt input
prompt = ChatPromptTemplate.from_template(
    """
    You are the writer of a University Project titled: 'STORAGE OF PERISHABLE FOOD USING AN IMPROVISED MODIFIED ATMOSPHERIC STORAGE METHOD'
    This final year thesis summaries a research project aimed at developing oxygen depriving methodology to kill micro-organisms that cause spoilage.

    I will give you a question and context from said thesis from which the answer should come from. Frame the answer however you like with whatever example is neccessary but the answer **must** come from the context I give you.
    If the context doesn't provide an answer to the question, **state clearly** that such information is out of the scope of this thesis.

    Context:
    {context}

    Question:
    {question}
    """
)

rag_chain = (
    {
        'context' : retrieving_function,
        'question' : RunnablePassthrough()
    }
    | prompt
    | model_querying
)

# Create a function taking input from the console to run rag function
feedback = ""

while True:
    print("Kindly input your question here: \n")
    feedback = input()

    if (feedback == 'exit'): break

    response = rag_chain.invoke(feedback)
    print('\n')
    print(Markdown(response.text))
    print('\n')
