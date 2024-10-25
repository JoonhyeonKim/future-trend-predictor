import os
from dotenv import load_dotenv
from openai import OpenAI
from crewai import Agent, Task
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

class NewsGeneratorAgent(Agent):
    def __init__(self):
        wikipedia = WikipediaAPIWrapper()
        tools = [
            Tool(
                name="Wikipedia",
                func=wikipedia.run,
                description="Useful for getting additional context on topics"
            )
        ]
        
        super().__init__(
            name="News Generator",
            role="Futuristic News Writer",
            goal="Generate specific and plausible future news articles based on identified trends",
            backstory="I am an AI agent with expertise in technology trends, market analysis, and creative writing. I specialize in extrapolating current trends into detailed future scenarios.",
            allow_delegation=False,
            verbose=True,
            tools=tools
        )
        
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_news(self, trends, keywords, timeframe="1 month", specificity="high"):
        future_news = []
        for trend, score in trends:
            context = self.tools[0].run(trend)
            prompt = f"""
            다음 트렌드와 키워드를 바탕으로 향후 {timeframe} 내에 일어날 수 있는 구체적이고 상세한 미래 뉴스 기사를 한국어로 작성해주세요:

            주요 트렌드: {trend} (관련성 점수: {score})
            관련 키워드: {', '.join(keywords)}

            위키피디아 컨텍스트: {context}

            지침:
            1. 제공된 트렌드와 키워드를 기사의 주요 내용으로 사용하세요.
            2. 위키피디아 컨텍스트를 활용하여 기사에 깊이와 정확성을 더하세요.
            3. 기사의 구체성을 {specificity}로 유지하며, 구체적인 예시와 현실적인 시나리오를 포함하세요.
            4. 기사를 헤드라인, 다음 {timeframe} 내의 구체적인 날짜, 본문 텍스트로 구성하세요.
            5. 트렌드와 키워드에 직접적으로 관련된 현실적이고 단기적인 발전에 집중하세요.

            이 기사는 현재 트렌드에 기반을 두되, 그 진화를 예측하는 진정한 미래 뉴스 기사처럼 읽혀야 합니다.
            """
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",  # Ensure this model can handle Korean text
                messages=[
                    {"role": "system", "content": "당신은 현재 트렌드와 그 단기적 영향에 대해 깊이 있는 지식을 가진 기술 산업 분석가이자 뉴스 작가입니다. 당신의 기사는 정확성, 구체성, 그리고 통찰력 있는 단기 예측으로 유명합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            article = response.choices[0].message.content.strip()
            future_news.append(article)
        return future_news

    @property
    def task(self):
        return Task(
            description="Generate detailed and plausible future news articles based on current trends, incorporating additional context and potential societal impacts",
            agent=self,
            expected_output="A list of well-crafted, specific future news articles, each exploring the potential developments and impacts of identified trends"
        )

    def run(self, trends, keywords):
        return self.generate_news(trends, keywords)
