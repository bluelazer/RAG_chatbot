from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量
import os
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings

persist_directory = 'para_db'

dir = "docs"
base_dir = dir # 文档的存放目录
documents = []
for file in os.listdir(base_dir):
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# 文本的分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50,separators="\n\n")
splits = text_splitter.split_documents(documents)


from langchain_huggingface.embeddings import HuggingFaceEmbeddings
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
#embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
embeddings = HuggingFaceEmbeddings(model_name="TencentBAC/Conan-embedding-v1",model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)

# vectorstore = Chroma.from_documents(collection_name="example_collection",
#     documents=splits,embedding=embeddings, persist_directory=persist_directory)

vectorstore = Chroma(collection_name="example_collection",persist_directory=persist_directory,embedding_function=embeddings,)
#create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.8},
)




# while True:
#     print("打印retriever的结果：\n",100*"*","\n")
#     query = input("retriever，请输入查询内容：\n")
#     if query == "exit":
#         break
#     res = retriever.invoke(query)
#     print(res)




# #Update Documents:
# updated_document = Document(
#     page_content="qux",
#     metadata={"bar": "baz"}
# )

# vectorstore.update_documents(ids=["1001"],documents=[updated_document])
# print("查询update documents的结果：\n",100*"*","\n")


# #Delete Documents:
# vectorstore.delete(ids=["1001"])


# #Search:
# query0 = input("按score搜索，请输入查询内容：\n")
# results = vectorstore.similarity_search(query=query0,k=1)
# print("打印搜索的结果：\n",100*"*","\n")
# for doc in results:
#     print(f"* {doc.page_content} [{doc.metadata}]")


# # #Search with filter
# # results = vectorstore.similarity_search(query="thud",k=2,filter={"baz": "bar"})
# # print("打印按过滤条件搜索的结果：\n",100*"*","\n")
# # for doc in results:
# #     print(f"* {doc.page_content} [{doc.metadata}]")


# #Search with score
# print("打印按score搜索的结果：\n",100*"*","\n")
# query1 = input("按score搜索，请输入查询内容：\n")
# results = vectorstore.similarity_search_with_score(query=query1,k=2)
# for doc, score in results:
#     print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

# #Add Documents:
# from langchain_core.documents import Document

# document_1 = Document(page_content="foo", metadata={"baz": "bar"})
# document_2 = Document(page_content="thud", metadata={"bar": "baz"})
# document_3 = Document(page_content="i will be deleted :(")

# documents = [document_1, document_2, document_3]
# ids = ["1000", "1001", "1002"]
# vectorstore.add_documents(documents=documents, ids=ids)
