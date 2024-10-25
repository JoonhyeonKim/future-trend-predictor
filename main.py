from scripts.news_fetcher import fetch_news
from scripts.news_indexer import index_news
from scripts.rag_agent import RAGAgent
from scripts.trend_analyzer import analyze_trends
from scripts.news_generator import NewsGeneratorAgent
from crewai import Crew, Process
from langgraph.graph import Graph

def main():
    # Step 1: Fetch news articles
    news_articles = fetch_news()

    # Step 2: Index news articles for RAG
    index_news(news_articles)

    # Step 3: Create RAG agent
    rag_agent = RAGAgent()

    # Step 4: Create trend analyzer
    trend_analyzer = analyze_trends

    # Step 5: Create news generator agent
    news_generator = NewsGeneratorAgent()

    # Step 6: Create a crew with the agents
    crew = Crew(
        agents=[rag_agent, news_generator],
        tasks=[
            rag_agent.task,
            news_generator.task
        ],
        process=Process.sequential
    )

    # Step 8: Execute the workflow
    result = crew.kickoff()

    # Step 9: Inspect tasks_output
    print(f"Tasks output (list): {result.tasks_output}")

    # Step 10: Extract articles from the list
    articles = []
    for task_output in result.tasks_output:
        # Inspect each element to see if it contains the "generate_news" key or relevant output
        print(f"Task output: {task_output}")

        # Assuming each task_output could be a dictionary or an object with an attribute "output"
        if isinstance(task_output, dict) and "generate_news" in task_output:
            articles.extend(task_output["generate_news"])
        elif hasattr(task_output, "generate_news"):
            articles.extend(getattr(task_output, "generate_news", []))
        elif isinstance(task_output, list):
            # If task_output is a list itself, add those directly (depending on the structure)
            articles.extend(task_output)

    # Print the articles if any were found
    if articles:
        for article in articles:
            print(article)
    else:
        print("No articles found in the expected structure.")

if __name__ == "__main__":
    main()