import os
from dotenv import load_dotenv
from crewai import Agent, Task
from chromadb import Client, Settings
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper

class RAGAgent(Agent):
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        if not serpapi_key:
            raise ValueError("SERPAPI_API_KEY not found in environment variables")
        
        search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="useful for when you need to answer questions about current events"
            )
        ]
        
        super().__init__(
            name="RAG Agent",
            role="Trend Analyzer and Information Retriever",
            goal="Retrieve and analyze relevant information from the news database and current events",
            backstory="I am an AI agent with extensive knowledge in data analysis and current affairs. I specialize in identifying trends and retrieving relevant information from various sources, including our news database and real-time search results.",
            allow_delegation=False,
            verbose=True,
            tools=tools
        )
        
        self._client = Client(Settings(persist_directory="./data/chroma_db"))
        self._collection = self._client.get_collection("news_articles")

    def retrieve_relevant_info(self, trends):
        query = " ".join([trend for trend, _ in trends])
        results = self._collection.query(query_texts=[query], n_results=5)
        return results["documents"][0]

    @property
    def task(self):
        return Task(
            description="Retrieve and analyze news articles based on identified trends, supplementing with current event information",
            agent=self,
            expected_output="A comprehensive summary of relevant information based on current trends, including both historical and real-time data"
        )

    def run(self, trends, keywords):
        relevant_info = self.retrieve_relevant_info(trends)
        analysis = f"Based on the trends {trends} and keywords {keywords}, here's the relevant information: {relevant_info}"
        return analysis
