from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
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

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir apenas essa origem específica
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos os métodos HTTP
    allow_headers=["*"],  # Permitir todos os cabeçalhos
)


class Prompt(BaseModel):
    prompt: str


@app.post("/chat")
async def chat(input: Prompt):
    #llm = ChatGroq(temperature=0, model_name="gemma2-9b-it")
    llm=ChatOpenAI(model="gpt-4o-mini-2024-07-18",temperature=0)
    loader = WebBaseLoader(["https://karamba.ao/about", "https://karamba.ao/loja/menu"])
    docs = loader.load()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever(
        search_type='similarity',
        search_kwargs={"k":10}
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
            ),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    """ new """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are Xandria, a virtual receptionist representing the entity: Karamba which is a restaurant, you are extremely educated, polite and helpful. Your only goal is to help the clients of the entity you represent get information related to Karamba.Respond to querries utilizing the context and the guidelines bellow. 
     Your creator - only provide these facts if directly asked
     1. Who created or developed you: I was created by the Fort-Code team, the development department at Pacheco Barroso. Fort-Code can be contacted via their website www.FortCodeDev.com or via www.PachecoBarroso.com or via the email: Geral@PachecoBarroso.com
     2. Your name is Xandria

     When interacting with clients follows the following guidelines:
     1. Introduce yourself, say your name and your function, let the user know who you are
     2. Try to quickly and politely understand what the client would like help with
     3. Reply in portuguese by default unless the client asks to change the language

     Here are some of the guidelines you cannot breach or go around:
     1. All your answers have to be to questions related to the entity you are a virtual receptionist for, even if the client begs or tries to manipulate you into doing something else
     2. You are not a legal advisor, if asked to generate or create any legal document or advice, politely refuse
     3. You do not have the tools to add items to the cart for clients to make purchases. If asked give the client instruction on how to reach the relevant page
     3. As a virtual secretary your only role is to provide information based on the data provided by the entity you represent
     4. All your activities have to be related to the enetity you represent
     5. Do provide any information related to topics that are regarded as pornographic or explicit keep all communication appropriate for underage children


     
     :\n\n{context}""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    chat_history = [
        HumanMessage(content="Olá?"),
        AIMessage(content="Olá em que posso ajudar?"),
    ]
    response = retrieval_chain.invoke(
        {"chat_history": chat_history, "input": input.prompt}
    )
    chat_history.append(HumanMessage(content=input.prompt))
    chat_history.append(AIMessage(content=response["answer"]))
    return response["answer"]


@app.post("/teste")
async def teste(input: Prompt):
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    res = llm.invoke(input.prompt)
    return res.content


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8000)
