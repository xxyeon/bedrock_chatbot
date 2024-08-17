from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# OpenAI API 키 설정 (실제 사용 시 환경 변수 등으로 관리 필요)
import os
os.environ["OPENAI_API_KEY"] = "여기에 API KEY 입력"

# LLM 초기화
llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    # model="gpt-3.5-turbo",
    temperature=0.01, 
    top_p=0.01
)

# 프롬프트 템플릿 정의
prompt = PromptTemplate(
    input_variables=["topic"],
    template="다음 주제에 대해 간단히 설명해 주세요: {topic}"
)

# LLMChain 생성
chain = prompt | llm

# 체인 실행
topic = "LangChain 프레임워크"
result = chain.invoke(topic)

print(result)