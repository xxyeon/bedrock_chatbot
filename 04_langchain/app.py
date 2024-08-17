import boto3
import streamlit as st
# For Claude 3, use BedrockChat instead of Bedrock
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Bedrock 클라이언트 설정
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Claude 3.5 파라미터 설정
model_kwargs =  { 
    "max_tokens": 1000,
    "temperature": 0.01,
    "top_p": 0.01,
}

# Bedrock LLM 설정
llm = ChatBedrock(
    client=bedrock_runtime,
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs=model_kwargs,
    streaming=True
)

# Streamlit 앱 설정
st.title("Chatbot powered by Bedrock and LangChain")

# Streamlit 채팅 메시지 히스토리 설정
message_history = StreamlitChatMessageHistory(key="chat_messages")

# 프롬프트 템플릿 설정
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="message_history"),
        ("human", "{query}"),
    ]
)

# 대화 체인 설정
chain_with_history = RunnableWithMessageHistory(
    prompt | llm,
    lambda session_id: message_history,  # 항상 이전 대화를 리턴
    input_messages_key="query",
    history_messages_key="message_history",
)

# 채팅 인터페이스
for msg in message_history.messages:
    st.chat_message(msg.type).write(msg.content)

# 사용자 입력 처리
if query := st.chat_input("Message Bedrock..."):
    st.chat_message("human").write(query)

    # chain이 호출되면 새 메시지가 자동으로 StreamlitChatMessageHistory에 저장됨
    config = {"configurable": {"session_id": "any"}}
    response_stream = chain_with_history.stream({"query": query},config=config)
    st.chat_message("ai").write_stream(response_stream)