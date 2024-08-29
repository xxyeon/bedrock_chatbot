## [Amazon Bedrock](https://docs.aws.amazon.com/ko_kr/bedrock/latest/userguide/what-is-bedrock.html)이란?
![image](https://github.com/user-attachments/assets/9e495768-ab05-462c-b01e-87091c34bbff)

- 완전관리형(a fully managed service == serverless)
- 주요 탑티어 AI 스타트업과 아마존의 파운데이션 모델(FMs)을 하나의 API로 사용
>[FM](https://aws.amazon.com/ko/what-is/foundation-models/)란   
>대규모 데이터세트를 기반으로 훈련된 파운데이션 모델(FM)은   
>데이터 사이언티스트가 기계 학습(ML)에 접근하는 방식을 변화시킨 대규모 딥 러닝 신경망입니다.    
>데이터 사이언티스트는 처음부터 인공 지능(AI)을 개발하지 않고 파운데이션 모델을 출발점으로 삼아 새로운 애플리케이션을 더 빠르고 비용 효율적으로 지원하는 ML 모델을 개발합니다.   

- 쉽게 말해서 AI에서 부족한 정보를 FM에서 가져다 사용한다고 생각하면 된다. (많은 데이터를 AI에게 주면서 학습시키는게 아니라 FM을 기반으로 학습)

## 실습환경 구성
![image](https://github.com/user-attachments/assets/7b92dbcd-0981-477e-8b93-c276f22cac0f)

### 1. cloud9 생성
- cloud9: 클라우드 기반 통합 개발 환경(IDE)
- EC2 + packages + Editor
### 2. Boto3 설치
- Boto3: Python SDK for AWS
  ```python
  pip install boto3
  ```
- AWS credential 필요한데 cloud9 환겨와 함께 생성되는 EC2 인스턴스인 경우 이미 설치되어 있다.
>bedrock 관련 기능이 있는 boto3 패키지를 사용하고 싶다면 boto3버전이 1.28.57인지 확인해봐야한다.
>```console
>pip freeze | grep boto
>#boto3 버전이 1.28.57 이상인지 확인
>```
![image](https://github.com/user-attachments/assets/b234c828-8940-49d7-8b2f-08daf11ae1d2)

### 3. streamlit 설치 및 실행
프론트엔드 구현에 도움을 주는 툴?이다. 컴포넌트를 그냥 가져다 쓰기만 하면된다.   
<br>
1. streamlit 설치
```python
pip install streamlit
```
2. streamlit 실행
```python
streamlit run {streamlit이 import된 python파일}
```

## AWS Bedrock 실습
### 1. 사용하고 싶은 모델 지정
```python
import boto3
# ... 생략 ...
# model을 지정하고, '그' 모델이 요구하는 body를 추가!
bedrock_runtime.invoke(modelId='xxx-claude3-xxx', body={ prompt: 'What is Amazon Bedrock' })
```
### 2. 챗봇에서 요청과 응답(JSON)
```python
# 유저의 질문(요청)을 JSON 형식으로 모델(gen AI)에게 보내기 위함
body = json.dumps({"inputText": question})  # 질문 텍스트를 JSON 형식으로 인코딩

# 모델에서 받아온 응답을 바디에서 결과를 추출하고 JSON으로 파싱
response_body = json.loads(response.get("body").read())
```
예시 코드
```python
def get_response(prompt):
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

        response = bedrock_runtime.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=body,
        )
        response_body = json.loads(response.get("body").read())
        output_text = response_body["content"][0]["text"]
        return output_text
    except Exception as e:
        print(e)
```
### 3. 생성형 AI 사용시 고려할 점
```python
#생성형 ai 사용시 고려 temperatur, top_p
body = json.dumps(
    {
        "prompt": question,
        "temperature": 0.5, #->0~1 창의적이게 하고 싶으면 temperature와 top_p올리고, 0에 가까울수록 사실적...
        "top_p": 0.5,  # vs. topP in titan -> 학습한 것중에서 상위 몇%만 본다. top_p가 줄어들수록 정확해짐
        "max_gen_len": 512,
    }
)
```

### 4. 유저와 대화 가능하도록 대화 맥락 기억하도록 구현
- 이전 대화에 사용자 입력을 추가하여 리스트로 전달
- 모델에서 처리할 수 있는 토큰 수 반드시 고려
예시 코드
```python
# 세션 상태에 저장된 메시지 순회하며 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # 채팅 메시지 버블 생성
        st.markdown(message["content"])  # 메시지 내용 마크다운으로 렌더링
...
# 사용자로부터 입력 받음
if prompt := st.chat_input("Message Bedrock..."):
    # 사용자 메시지를 세션 상태에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):  # 사용자 메시지 채팅 메시지 버블 생성
        st.markdown(prompt)  # 사용자 메시지 표시
    # 보조 응답을 세션 상태에 추가
    output_text = get_response(prompt) #ai의 답변
    st.session_state.messages.append({"role": "assistant", "content": output_text}) # 보조 메시지 채팅 메시지 버블 생성
    with st.chat_message("assistant"):
        st.markdown(output_text)

```
### 5. streaming
- 사용자게에 콘텐츠 즉시 반환하고 싶을 때 유용
- 전체 응답이 생성될 때가지 기다리는 것은 사용자 경험에 부정적
- yield 개념을 알아야함. 간단히 설명 후 넘어감

### open AI 사용법
```python
llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    # model="gpt-3.5-turbo",
    temperature=0.01, 
    top_p=0.01
)
```
