# Future Trend Predictor

This project is designed to analyze current news data and predict future trends using an LLM-based agent. It extracts keywords from recent news articles and generates potential future scenarios using the GPT API.

## Features
- **Keyword Extraction**: Analyzes news articles to identify trending topics.
- **Trend Analysis**: Uses LangGraph to refine the understanding of emerging patterns.
- **Future News Generation**: Generates future news stories based on detected trends using GPT.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/future-trend-predictor.git
   ```
2. Navigate into the project directory:
   ```bash
   cd future-trend-predictor
   ```
3. Set up a virtual environment and install dependencies:
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```
4. Set the API keys in the .env file
   ```bash
   touch .env
   ```
   OPENAI_API_KEY=your_openai_key_here
   NEWS_API_KEY=your_news_api_key_here
   SERPAPI_API_KEY=your_serpapi_key_here

## Usage
To run the main script:
```bash
python main.py
```
## License
MIT License

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Contact
For any inquiries, please contact joonhyeonkim1015@gmail.com.
