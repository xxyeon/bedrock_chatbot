import json

import boto3

# Bedrock runtime
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")


def chunk_handler(chunk):
    #  API가 서로 다른 타입을 리턴
    # print(f"\n\n!!!\n{chunk}")
    text = ""
    chunk_type = chunk.get("type")
    # print(f"\n\nchunk type: {chunk_type}")
    if chunk_type == "message_start":
        # 첫 번째 청크는 message role에 대한 정보를 포함
        role = chunk["message"]["role"]
        text = ""
    elif chunk_type == "content_block_start":
        # 응답 텍스트 시작
        text = chunk["content_block"]["text"]
    elif chunk_type == "content_block_delta":
        # 스트리밍 중인 응답 텍스트의 일부
        text = chunk["delta"]["text"]
    elif chunk_type == "message_delta":
        # 응답이 중단되거나 완료된 이유를 포함
        stop_reason = chunk["delta"]["stop_reason"]
        text = ""
    elif chunk_type == "message_stop":
        # 요청에 대한 메트릭을 포함
        metric = chunk["amazon-bedrock-invocationMetrics"]
        inputTokenCount = metric["inputTokenCount"]
        outputTokenCount = metric["outputTokenCount"]
        firstByteLatency = metric["firstByteLatency"]
        invocationLatency = metric["invocationLatency"]
        text = ""

    print(text, end="")
    return text


def get_streaming_response(prompt, streaming_callback):
    try:
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

        # stream
        response = bedrock_runtime.invoke_model_with_response_stream(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            # modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            body=body,
        )
        stream = response.get("body")

        if stream:
            for event in stream:  # 스트림에서 반환된 각 이벤트 처리
                chunk = event.get("chunk")
                if chunk:
                    chunk_json = json.loads(chunk.get("bytes").decode())
                    dstreaming_callback(chunk_json)
    except Exception as e:
        print(e)


get_streaming_response("대한민국의 수도에 대해 설명해", chunk_handler)
