import json

import boto3
import streamlit as st

bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

st.title("Chatbot powered by Bedrock")

# 새로고침 , 새로 접속
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Message Bedrock...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # LLM에 prompt를 보낸다
    # Amazon Bedrock을 사용할 것
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }
    )

    response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=body,
    )
    response_body = json.loads(response.get("body").read())
    output_text = response_body["content"][0]["text"]
    st.session_state.messages.append({"role": "assistant", "content": output_text})
    with st.chat_message("assistant"):
        st.markdown(output_text)
