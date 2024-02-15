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

# list all FMs
# fms = bedrock.list_foundation_models()
# printJson(fms)

titan = bedrock.get_foundation_model(modelIdentifier="amazon.titan-tg1-large")
# printJson(titan)
# 참고: Reference '[201] 모델 예시' 참고

# Bedrock runtime
# boto3.client(service_name='bedrock'): bedrock에 대한 정보
# bedrock을 통해 모델을 실행하려면 'bedrock-runtime' 서비스를 이용
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
question = "What is the AWS Bedrock?"
body = json.dumps({"inputText": question})
response = bedrock_runtime.invoke_model(body=body, modelId="amazon.titan-tg1-large")
# print(response)
# 'body': <botocore.response.StreamingBody object at 0x7f067f91c7f0>
response_body = json.loads(response.get("body").read())
# printJson(response_body)
# print(response_body['results'][0]['outputText'])


# 모델별 적절한 파라미터 전달
# 참고: Reference '[202] Bedrock runtime 파라미터' 참고
body = json.dumps(
    {
        "inputText": question,
        "textGenerationConfig": {
            "temperature": 0.5,
            "topP": 0.5,
            "maxTokenCount": 512,
            "stopSequences": ["something"],
        },
    }
)


# temperature and top p
body = json.dumps(
    {
        "inputText": question,
        "textGenerationConfig": {
            "temperature": 0,
            "topP": 0.01,
            "maxTokenCount": 512,
        },
    }
)

response = bedrock_runtime.invoke_model(body=body, modelId="amazon.titan-tg1-large")
response_body = json.loads(response.get("body").read())
print(response_body["results"][0]["outputText"])

response = bedrock_runtime.invoke_model(body=body, modelId="amazon.titan-tg1-large")
response_body = json.loads(response.get("body").read())
print(response_body["results"][0]["outputText"])
