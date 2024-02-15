import json


def printJson(dataObj):
    print(json.dumps(dataObj, sort_keys=True, indent=4))


import boto3

# access key id와 secret key가 필요
# bedrock = boto3.client(aws_access_key_id='INPUT YOUR KEY',
#                       aws_secret_access_key='INPUT YOUR KEY')

# Cloud9 EC2는 credential이 설정되어 있으므로 생략 가능
# bedrock을 지원하는 리전을 일부 + 각 모델별 access 요청 필요
bedrock = boto3.client(service_name="bedrock", region_name="us-east-1")

# Bedrock runtime
# boto3.client(service_name='bedrock'): bedrock에 대한 정보
# bedrock을 통해 모델을 실행하려면 'bedrock-runtime' 서비스를 이용
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
question = "What is the AWS Bedrock?"
body = json.dumps({"prompt": question})
response = bedrock_runtime.invoke_model(body=body, modelId="meta.llama2-13b-chat-v1")
# print(response)
# 'body': <botocore.response.StreamingBody object at 0x7f067f91c7f0>
response_body = json.loads(response.get("body").read())
# printJson(response_body)
print(response_body["generation"])

# 모델별 적절한 파라미터 전달
# 참고: Reference '[202] Bedrock runtime 파라미터' 참고
body = json.dumps(
    {
        "prompt": question,
        "temperature": 0.5,
        "top_p": 0.5,  # vs. topP in titan
        "max_gen_len": 512,
    }
)

# llama2: Meta가 개발한 오픈 소스 모델
# 일부 상업적 목적 사용(commercial usee)에 제한이 있음
# 월 사용자 7억명 이상(+700M MAU)일 경우 Meta에 별도의 라이센스 요청 필요
# 경쟁자 견제 정도로 보임
