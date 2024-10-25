from crewai import Agent, Task
from chromadb import Client, Settings
# from pydantic import Field

class RAGAgent(Agent):
    def __init__(self):
        super().__init__(
            name="RAG Agent",
            role="Trend Analyzer",
            goal="Retrieve relevant information from the news database",
            backstory="I am an AI agent specialized in retrieving and analyzing news data.",
            allow_delegation=False
        )
        
        # Initialize private attributes for the client and collection
        self._client = Client(Settings(persist_directory="./data/chroma_db"))
        self._collection = self._client.get_collection("news_articles")

    def retrieve_relevant_info(self, query):
        results = self.collection.query(query_texts=[query], n_results=5)
        return results["documents"][0]

    @property
    def task(self):
        return Task(
            description="Retrieve and analyze news articles to identify trends",
            agent=self,
            expected_output="A list of key trends extracted from recent news articles"  # Add the expected output
        )

    def run(self, context=None):
        query = "Latest technology trends and their potential future impact"
        relevant_info = self.retrieve_relevant_info(query)
        return "\n".join(relevant_info)
    