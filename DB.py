from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "notes.txt", "topic": "Python"}
    ),
    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "notes.txt", "topic": "Pandas"}
    ),
    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "notes.txt", "topic": "Deep Learning"}
    ),
]

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

result = vectorstore.similarity_search("What is used for data analysis?",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vectorstore.as_retriever()

docs = retriver.invoke("Explain")