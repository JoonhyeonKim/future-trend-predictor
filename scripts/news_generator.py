import os
from dotenv import load_dotenv
from openai import OpenAI
from crewai import Agent, Task

load_dotenv()

class NewsGeneratorAgent(Agent):
    def __init__(self):
        super().__init__(
            name="News Generator",
            role="News Writer",
            goal="Generate future news articles based on identified trends",
            backstory="I am an AI agent specialized in writing future news based on trends.",
            allow_delegation=False
        )
        
        # Change client to a private attribute
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_news(self, trends):
        future_news = []
        for trend, _ in trends:
            prompt = f"Write a future news article about the trend '{trend}' and its potential impact in the next year."
            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a futuristic news writer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )
            article = response.choices[0].message.content.strip()
            future_news.append(article)
        return future_news

    @property
    def task(self):
        return Task(
            description="Generate future news articles based on current trends",
            agent=self,
            expected_output="A list of future news articles generated based on the identified trends."
        )

    def run(self, trends):
        return self.generate_news(trends)