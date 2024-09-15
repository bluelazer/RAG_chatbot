from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量
from classes import get_retriever
from langchain_openai import ChatOpenAI
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

# # Helper function for printing docs


# def pretty_print_docs(docs):
#     print(
#         f"\n{'-' * 100}\n".join(
#             [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
#         )
#     )

# while True:
#     user_input = input("你: ")
#     doc = retriever.invoke(user_input)
#     compressed_docs = compression_retriever.invoke(user_input)
#     print(f"未压缩召回文档:\n")
#     print(doc)
#     print(100*"*")
#     print(f"压缩后的召回文档:\n")
#     pretty_print_docs(compressed_docs)