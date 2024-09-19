from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量
import os
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from langchain_community.document_loaders import DirectoryLoader

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量
import os
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings

persist_directory = 'rerankertestdb'

dir = "./"
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
        loader = TextLoader(file_path,encoding='utf-8')
        documents.extend(loader.load())

# 文本的分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0,separators="\n\n")
splits = text_splitter.split_documents(documents)


from langchain_huggingface.embeddings import HuggingFaceEmbeddings
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
#embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
embeddings = HuggingFaceEmbeddings(model_name="TencentBAC/Conan-embedding-v1",model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)

# vectorstore = Chroma.from_documents(collection_name="example_collection",
#     documents=splits,embedding=embeddings, persist_directory=persist_directory)

vectorstore = Chroma.from_documents(collection_name="example_collection1",
    documents=splits,embedding=embeddings, persist_directory=persist_directory)
#create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.8},
)


#初始化reranker模型
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


#定义reranker函数
def rerank(query,documents):
    """使用bge-reranker对检索结果进行重排"""
    document_texts = [doc.page_content for doc in documents]
    pairs = [[x,y] for x,y in zip(document_texts,[str(query)]*len(document_texts))]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(100*"=","\n重排序分数是：\n",scores)
    ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked_docs]

#定义查询处理流程
def handle_query(query):
    #初始检索
    results = retriever.get_relevant_documents(query)
    #重排
    ranked_results = rerank(query,results)
    return ranked_results


#示例查询
query = "中央集权"
print("重排前的文档顺序：")
unreanked_results = retriever.get_relevant_documents(query)
print(unreanked_results)

response = handle_query(query)
print("重排后的文档顺序：")
print(response)



