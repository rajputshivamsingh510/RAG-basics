#-----------USING CHARACTER TEXT SPLITTER

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter( seperator ="",
                                  chunk_size = 10,
                                 chunk_overlap = 0)

data = TextLoader("document loaders/notes.txt")

docs = data.load()

chunk = splitter.split_documents(docs)

print(chunk[0].page_content)
print(len(chunk))



