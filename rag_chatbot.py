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
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



### 添加记忆功能 ###
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}  # memory is maintained outside the chain
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

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
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",)


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
        # 调用 Retrieval Chain  
        response = qa.invoke({"input":user_input}, config={"configurable": {"session_id": session_id}})
        print(f"Chatbot: {response['answer']}")


# 启动 Chatbot
if __name__ == "__main__":
    # 启动Chatbot
    chat_loop()