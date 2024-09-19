from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def get_retriever(model_name,persist_directory,collection_name):
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    #embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
    embeddings = HuggingFaceEmbeddings(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)

    persist_directory = persist_directory
    vectorstore = Chroma(collection_name=collection_name,persist_directory=persist_directory,embedding_function=embeddings,)
    #create retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.8},
    )

    return retriever

def get_compression_retriever():
    persist_directory = 'para_db'
    collection_name="example_collection"
    model_name="TencentBAC/Conan-embedding-v1"
    retriever = get_retriever(model_name=model_name, persist_directory=persist_directory, collection_name=collection_name)



    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor

    llm = ChatOpenAI(model = "gpt-4o-mini",max_tokens=3000)
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever

def rerank(query, retieved_docs,modelid="BAAI/bge-reranker-base"):
    tokenizer = AutoTokenizer.from_pretrained(modelid)
    model = AutoModelForSequenceClassification.from_pretrained(modelid)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    document_texts = [doc.page_content for doc in retieved_docs]
    pairs = [[x,y] for x,y in zip(document_texts,[str(query)]*len(document_texts))]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(100*"=","\n重排序分数是：\n",scores)
    ranked_docs = sorted(zip(retieved_docs, scores), key=lambda x: x[1], reverse=True)
    return ranked_docs
