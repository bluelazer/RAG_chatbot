from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

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