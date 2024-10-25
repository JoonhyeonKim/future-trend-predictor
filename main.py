from scripts.news_fetcher import fetch_news
from scripts.news_indexer import index_news
from scripts.rag_agent import RAGAgent
from scripts.trend_analyzer import analyze_trends
from scripts.keyword_extractor import extract_keywords
from scripts.news_generator import NewsGeneratorAgent
from crewai import Crew, Process, Task
import numpy as np  # Add this import at the top of the file

def main():
    # Step 1: Fetch news articles
    news_articles = fetch_news()

    if not news_articles:
        print("No news articles fetched. Exiting.")
        return

    # Step 2: Extract keywords
    keywords = extract_keywords(news_articles)

    if len(keywords) == 0:
        print("No keywords extracted. Exiting.")
        return

    # Step 3: Analyze trends
    trends = analyze_trends(" ".join(keywords))

    if not trends:
        print("No trends identified. Exiting.")
        return

    # Step 4: Index news articles for RAG
    index_news(news_articles)

    # Convert NumPy arrays to lists if necessary
    keywords = keywords.tolist() if isinstance(keywords, np.ndarray) else keywords
    trends = [tuple(t) if isinstance(t, np.ndarray) else t for t in trends]

    print("Extracted Keywords:", keywords)
    print("Identified Trends:", trends)

    # Step 5: Create RAG agent
    rag_agent = RAGAgent()

    # Step 6: Create news generator agent
    news_generator = NewsGeneratorAgent()

    # Step 7: Create a crew with the agents
    crew = Crew(
        agents=[rag_agent, news_generator],
        tasks=[
            Task(
                description="Analyze trends and retrieve relevant information",
                agent=rag_agent,
                expected_output="A detailed analysis of current trends based on the news database and real-time search results."
            ),
            Task(
                description="Generate future news articles based on trends and analysis",
                agent=news_generator,
                expected_output="A list of well-crafted, specific future news articles, each exploring the potential developments and impacts of identified trends."
            )
        ],
        process=Process.sequential
    )

    # Step 8: Execute the workflow
    result = crew.kickoff(inputs={"trends": trends, "keywords": keywords})

    # Step 9: Print the results
    print("Extracted Keywords:", keywords)
    print("Identified Trends:", trends)
    print("Generated Future News:")
    for task_output in result.tasks_output:
        print(task_output)

if __name__ == "__main__":
    main()
