from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

store = {}  # memory is maintained outside the chain
llm = ChatOpenAI(model="gpt-4o-mini")
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = {
            "chat_history": InMemoryChatMessageHistory(),
            "summary_memory": ConversationSummaryBufferMemory(
                chat_memory=InMemoryChatMessageHistory(),
                llm=llm,
                max_token_limit=100,  # 你可以根据需要调整这个值
                return_messages=True,
            )
        }

    # 获取当前的聊天历史和摘要记忆
    chat_history = store[session_id]["chat_history"]
    summary_memory = store[session_id]["summary_memory"]

    # 更新摘要记忆中的聊天历史
    summary_memory.chat_memory = chat_history

    # 加载记忆变量
    assert len(summary_memory.memory_variables) == 1
    key = summary_memory.memory_variables[0]
    messages = summary_memory.load_memory_variables({})[key]

    # 更新聊天历史
    store[session_id]["chat_history"] = InMemoryChatMessageHistory(messages=messages)
    print(100*"=","\n下面是记忆的内容：\n",store[session_id]["chat_history"])

    return store[session_id]["chat_history"]



chain = RunnableWithMessageHistory(llm, get_session_history)

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    res = chain.invoke(user_input,config={"configurable": {"session_id": "abc123"}}).content
    print(100*"=","\n下面是机器的回答：\n")
    print(res)  
