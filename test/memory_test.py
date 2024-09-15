from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI



llm = ChatOpenAI(model = "gpt-4o-mini",max_tokens=3000)

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



chain = RunnableWithMessageHistory(llm, get_session_history)

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    res = chain.invoke(user_input,config={"configurable": {"session_id": "abc123"}}).content
    print(100*"=","\n下面是机器的回答：\n")
    print(res)  
