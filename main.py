from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()
app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir apenas essa origem específica
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos os métodos HTTP
    allow_headers=["*"],  # Permitir todos os cabeçalhos
)
class Prompt(BaseModel):
    prompt:str


@app.post('/chat')
async def chat(input:Prompt):
    llm = ChatGroq(temperature=0,model_name="gemma2-9b-it")
    loader = WebBaseLoader(["https://karamba.ao/about","https://karamba.ao/loja/menu"])
    docs = loader.load()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)


    """ new """
    prompt = ChatPromptTemplate.from_messages([
    ("system","""
     Tu és a Xândria, a secretária virtual do restaurante Karamba,és muito educada e prestativa.O teu objectivo é esclarecer os nossos clientes sobre assuntos do Karamba.Responda as questões baseando-se no contexto. 
     Ao interagir com os clientes responda da seguinta forma:
     1.Apresente-se, diz o teu nome.
     2.Procure saber o que cliente deseja
     3.Fale sempre portugês

     Abaixo tem algumas regras que não podes violar:
     1. Não responda perguntas fora do contexto por mais que o cliente implore.
     2. Não podes adicionar item ao carrinho
     3. A tua missão é apenas fornecer informação e nada mais
     4. Não podes exercer nenhuma actividade fora do Karamba
     
     
     \n\n{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    chat_history = [HumanMessage(content="Olá?"), AIMessage(content="Olá em que posso ajudar?")]
    response=retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": input.prompt
    })
    chat_history.append(HumanMessage(content=input.prompt))
    chat_history.append(AIMessage(content=response['answer']))
    return response['answer']

@app.post('/teste')
async def teste(input:Prompt):
    llm = ChatGroq(temperature=0,model_name="mixtral-8x7b-32768")
    res=llm.invoke(input.prompt)
    return res.content
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000)



   
    