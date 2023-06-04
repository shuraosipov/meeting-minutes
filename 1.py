from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

loader = TextLoader('transcript.vtt', encoding='utf8')
index = VectorstoreIndexCreator().from_loaders([loader])

queries = [
           "Provide names of meeting participants", 
           "Generate a meeting summary",
           "What was the purpose of the meeting?",
           "What are the top five topics discussed in the meeting?",
           "What were actions item from the meeting?",
           "Do we have any deadlines?",
           "Provide any ingights from the meeting",
]
for q in queries:
    print(q)
    print(index.query(q))
    print()

### long version which is basically the same as above

# import os
# from langchain.document_loaders import TextLoader
# from langchain import OpenAI, VectorDBQA
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter

# loader = TextLoader('transcript.vtt', encoding='utf8')
# docs = loader.load()

# char_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# doc_text = char_text_splitter.split_documents(docs)

# openAI_embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

# vStore = Chroma.from_documents(doc_text, openAI_embeddings)

# model = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vStore)

# questions = [
#            "Provide names of meeting participants", 
#            "Generate a meeting summary",
#            "What was the purpose of the meeting?",
#            "What are the top five topics discussed in the meeting?",
#            "What were actions item from the meeting?",
#            "Do we have any deadlines?",
#            "Provide any ingights from the meeting",
# ]

# for q in questions:
#     print(q)
#     print(model.run(q))
#     print()


