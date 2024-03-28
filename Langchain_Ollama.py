from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import HuggingFaceEmbeddings

# Create an instance of HuggingFaceEmbeddings for text embeddings
embedding = HuggingFaceEmbeddings()

# Create a WebBaseLoader to load data from a web page
loader = WebBaseLoader("https://oldschoolrunescape.fandom.com/wiki/Fire_cape")
data = loader.load()

# Create a RecursiveCharacterTextSplitter to split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=["\n","\r\n","\n\n"," ",".",","])
all_splits = text_splitter.split_documents(data)

# Create a DocArrayInMemorySearch from the document splits and the embedding
db = DocArrayInMemorySearch.from_documents(all_splits, embedding)

# Create an instance of Ollama for language model-based retrieval
ollama = Ollama(base_url='http://127.0.0.1:11434', model='gemma:2b', temperature=0)

# Create a RetrievalQA instance using the Ollama language model, the document retriever, and enable verbose mode
qa = RetrievalQA.from_chain_type(
    llm=ollama,
    retriever=db.as_retriever(),
    verbose=True
)

# Run the question-answering process on the given query
response = qa.run("Explain me the process of getting a fire cape in old school runescape")
print(response)





