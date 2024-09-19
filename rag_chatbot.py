from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量
from classes import get_compression_retriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

import langchain
#langchain.debug = True


persist_directory = 'para_db'
collection_name="example_collection"
model_name="TencentBAC/Conan-embedding-v1"
retriever = get_compression_retriever()

llm = ChatOpenAI(model = "gpt-4o-mini",max_tokens=3000)

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = (
    "你是一个回答问题的助手。"
    "使用以下提供的上下文信息来回答问题。如果你不知道答案，就说你不知道。"
    "最多使用三句话，并保持答案简洁。"
    "\n\n上下文信息如下：\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

#rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



### 添加记忆功能 ###
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}  # memory is maintained outside the chain
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    memory = ConversationSummaryBufferMemory(
        chat_memory=store[session_id],
        llm=llm,
        max_token_limit=100,  # Adjust this according to your needs
        return_messages=True,
    )
    memory.prune()
    assert len(memory.memory_variables) == 1
    key = memory.memory_variables[0]
    messages = memory.load_memory_variables({})[key]
    store[session_id] = InMemoryChatMessageHistory(messages=messages)
    print(100*"=","\n下面是历史对话：\n",store[session_id])
    return store[session_id]

qa = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        #output_messages_key="answer",
        )

from classes import rerank
# 交互对话的函数
def chat_loop():
    print("Chatbot 已启动! 输入'exit'来退出程序。")
    while True:
        session_id = "abc123"
        user_input = input("你: ")
        if user_input.lower() == 'exit':
            print("再见!")
            break
        if user_input.lower() == 'clear':
            store[session_id] = InMemoryChatMessageHistory()
            print("对话历史已清空。")
            continue
        
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
            chat_history=[]
        else:
            # 获取对话历史
            chat_history = store[session_id].messages

        retrieved_docs = history_aware_retriever.invoke({"input": user_input,"chat_history": chat_history})
        
        unique_docs = []
        for item in retrieved_docs:
            if item not in unique_docs:
                unique_docs.append(item)

        print(100*"*","\n检索到的文档：\n",unique_docs)
        if len(unique_docs) == 0:
            print("没有找到相关文档.")
            reranked_docs=[]
        else:
            reranked_docs = rerank(user_input,unique_docs)
            reranked_docs = [doc[0] for doc in reranked_docs]
        print(100*"*","\n重排序的文档：\n",reranked_docs)

        # 调用 Retrieval Chain  
        response = qa.invoke({"input":user_input,"context":reranked_docs}, config={"configurable": {"session_id": session_id}})
        print(100*"=","\n下面是AI的回答：\n")
        print(f"Chatbot: {response}")
        print(100*"=","\n")


# 启动 Chatbot
if __name__ == "__main__":
    # 启动Chatbot
    chat_loop()