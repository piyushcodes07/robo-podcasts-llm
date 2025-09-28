from concurrent.futures import ThreadPoolExecutor, as_completed

"""
Research module for podcast generation.

This module provides functionality to gather background research and information
for podcast episode generation. It handles retrieving content from various sources
like Wikipedia and search engines.

Example:
    >>> from podcast_llm.research import suggest_wikipedia_articles
    >>> from podcast_llm.models import WikipediaPages
    >>> config = PodcastConfig()
    >>> articles: WikipediaPages = suggest_wikipedia_articles(config, "Artificial Intelligence")
    >>> print(articles.pages[0].name)
    'Artificial intelligence'

The research process includes:
- Suggesting relevant Wikipedia articles via LangChain and GPT-4
- Downloading Wikipedia article content
- Performing targeted web searches with Tavily
- Extracting key information from web articles
- Organizing research into structured formats using Pydantic models

The module uses various APIs and services to gather comprehensive background
information while maintaining rate limits and handling errors gracefully.
"""


import logging
from typing import List, Optional
from langchain import hub
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.documents import Document
from podcast_llm.outline import PodcastOutline
from tavily import TavilyClient
from podcast_llm.config import PodcastConfig
from podcast_llm.utils.llm import get_fast_llm
from podcast_llm.models import SearchQueries, WikipediaPages
from podcast_llm.extractors.web import WebSourceDocument
from concurrent.futures import ThreadPoolExecutor, as_completed
from podcast_llm.streamer import streamer

logger = logging.getLogger(__name__)


def suggest_wikipedia_articles(config: PodcastConfig, topic: str) -> WikipediaPages:
    """
    Suggest relevant Wikipedia articles for a given topic.

    Uses LangChain and GPT-4 to intelligently suggest Wikipedia articles that would provide good
    background research for a podcast episode on the given topic.

    Args:
        topic (str): The podcast topic to research

    Returns:
        WikipediaPages: A structured list of suggested Wikipedia article titles
    """
    logger.info(f"Suggesting Wikipedia articles for topic: {topic}")

    prompthub_path = "evandempsey/podcast_wikipedia_suggestions:58c92df4"
    wikipedia_prompt = hub.pull(prompthub_path)
    logger.info(f"Got prompt from hub: {prompthub_path}")

    fast_llm = get_fast_llm(config)
    wikipedia_chain = wikipedia_prompt | fast_llm.with_structured_output(WikipediaPages)
    result = wikipedia_chain.invoke({"topic": topic})
    logger.info(f"Found {len(result.pages)} suggested Wikipedia articles")
    return result


async def download_wikipedia_articles(
    suggestions: WikipediaPages, user_id: Optional[str] = None
) -> list:
    """
    Download Wikipedia articles in parallel based on suggested page titles.
    """

    logger.info("Starting Wikipedia article download")
    retriever = WikipediaRetriever()

    def fetch_article(page_name: str):
        logger.info(f"Retrieving article: {page_name}")

        try:
            return retriever.invoke(page_name)[0]
        except Exception as e:
            logger.error(f"Failed to retrieve article {page_name}: {str(e)}")
            return None

    wikipedia_documents = []
    total_articles = len(suggestions.pages)
    downloaded_count = 0

    with ThreadPoolExecutor(
        max_workers=min(32, total_articles)
    ) as executor:  # tune max_workers as needed
        future_to_page = {
            executor.submit(fetch_article, page.name): page
            for page in suggestions.pages
        }

        for future in as_completed(future_to_page):
            page = future_to_page[future]
            try:
                result = future.result()
                if result:
                    wikipedia_documents.append(result)
                    downloaded_count += 1
                    if user_id:
                        progress = int((downloaded_count / total_articles) * 100)
                        asyncio.create_task(
                            streamer.send(
                                user_id,
                                "Research",
                                progress,
                                f"Downloaded Wikipedia article: {page.name}",
                            )
                        )
                    logger.debug(f"Successfully retrieved article: {page.name}")
            except Exception as e:
                logger.error(f"Error retrieving {page.name}: {str(e)}")

    logger.info(
        f"Downloaded {len(wikipedia_documents)} Wikipedia articles (out of {total_articles})"
    )
    return wikipedia_documents


async def research_background_info(
    config: PodcastConfig, topic: str, user_id: Optional[str] = None
) -> list:
    """
    Research background information for a podcast topic.

    Coordinates the research process by first suggesting relevant Wikipedia articles
    based on the topic, then downloading the full content of those articles. Acts as
    the main orchestration function for gathering background research material.

    Args:
        topic (str): The podcast topic to research

    Returns:
        dict: List of retrieved Wikipedia document objects containing article content and metadata
    """
    logger.info(f"Starting research for topic: {topic}")
    if user_id:
        await streamer.send(user_id, "Research", 10, "Suggesting Wikipedia articles...")

    suggestions = suggest_wikipedia_articles(config, topic)
    if user_id:
        await streamer.send(
            user_id,
            "Research",
            30,
            f"Found {len(suggestions.pages)} Wikipedia articles. Downloading...\n {str(suggestions)}",
        )

    wikipedia_content = await download_wikipedia_articles(suggestions, user_id)

    logger.info("Research completed successfully")
    return wikipedia_content


def perform_tavily_queries(config: PodcastConfig, queries: SearchQueries) -> list:
    logger.info("Performing search queries")
    tavily_client = TavilyClient(api_key=config.tavily_api_key)

    exclude_domains = [
        "wikipedia.org",
        "youtube.com",
        "books.google.com",
        "academia.edu",
        "washingtonpost.com",
    ]

    urls_to_scrape = set()

    def search_query(query_str):
        try:
            response = tavily_client.search(
                query_str, exclude_domains=exclude_domains, max_results=5
            )
            return [
                r["url"] for r in response["results"] if not r["url"].endswith(".pdf")
            ]
        except Exception as e:
            logger.error(f"Error searching {query_str}: {e}")
            return []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(search_query, q.query) for q in queries.queries]
        for future in as_completed(futures):
            urls_to_scrape.update(future.result())

    return list(urls_to_scrape)


def download_page_content(urls: List[str]) -> List[Document]:
    logger.info("Downloading page content from URLs.")
    downloaded_articles = []

    def fetch_and_parse(url):
        try:
            web_source_doc = WebSourceDocument(url)
            web_source_doc.extract()
            return web_source_doc.as_langchain_document()
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return None

    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(fetch_and_parse, url) for url in urls]
        for future in as_completed(futures):
            doc = future.result()
            if doc:
                downloaded_articles.append(doc)

    logger.info(f"Successfully downloaded {len(downloaded_articles)} articles")
    return downloaded_articles


def research_discussion_topics(
    config: PodcastConfig, topic: str, outline: PodcastOutline
) -> list:
    """
    Research in-depth content for podcast discussion topics.

    Takes a podcast topic and outline, then uses LangChain and GPT-4 to generate targeted
    search queries. These queries are used to find relevant articles via Tavily search.
    The articles are then downloaded and processed to provide detailed research material
    for each section of the podcast.

    Args:
        topic (str): The main topic for the podcast episode
        outline (PodcastOutline): Structured outline containing sections and subsections

    Returns:
        list: List of dictionaries containing downloaded article content with structure:
            {
                'url': str,      # Source URL
                'title': str,    # Article title
                'text': str      # Article content
            }
    """
    logger.info(f"Suggesting search queries based on podcast outline")
    prompthub_path = "evandempsey/podcast_research_queries:561acf5f"

    search_queries_prompt = hub.pull(prompthub_path)
    logger.info(f"Got prompt from hub: {prompthub_path}")

    fast_llm = get_fast_llm(config)
    search_queries_chain = search_queries_prompt | fast_llm.with_structured_output(
        SearchQueries
    )
    queries = search_queries_chain.invoke(
        {"topic": topic, "podcast_outline": outline.as_str}
    )
    logger.info(f"Got {len(queries.queries)} suggested search queries")

    urls_to_scrape = perform_tavily_queries(config, queries)
    page_content = download_page_content(urls_to_scrape)
    return page_content
