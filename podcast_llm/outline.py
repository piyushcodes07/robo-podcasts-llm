"""
Podcast outline generation module.

This module provides functionality for generating and structuring podcast outlines.
It contains utilities for formatting and manipulating outline structures, as well as
functions for generating complete podcast outlines from topics and research material.

The module leverages LangChain and GPT-4 to intelligently structure podcast content
into a hierarchical outline format. It uses prompts from the LangChain Hub to ensure
consistent and high-quality outline generation.

Functions:
    format_wikipedia_document: Formats Wikipedia content for use in prompts
    summarize_background_chunk: Summarizes batches of background docs (map step)
    outline_episode: Generates a complete podcast outline from topic + research (map–reduce)

Example:
    outline = outline_episode(
        config=podcast_config,
        topic="Artificial Intelligence",
        background_info=research_docs
    )
    print(outline.as_str)
"""

import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor
from langchain import hub
from podcast_llm.config import PodcastConfig
from podcast_llm.utils.llm import get_long_context_llm, get_fast_llm
from podcast_llm.models import PodcastOutline


logger = logging.getLogger(__name__)


def format_wikipedia_document(doc):
    """
    Format a Wikipedia document for use in prompt context.

    Takes a Wikipedia document object and formats its metadata and content into a
    structured string format suitable for inclusion in LLM prompts. The format
    includes a header with the article title followed by the full article content.

    Args:
        doc: Wikipedia document object containing metadata and page content

    Returns:
        str: Formatted string with article title and content
    """
    return f"### {doc.metadata['title']}\n\n{doc.page_content}"


def summarize_background_chunk(config: PodcastConfig, topic: str, docs: List) -> str:
    """
    Summarize a batch of background documents into a concise context string.

    Args:
        config (PodcastConfig): Podcast configuration with LLM settings
        topic (str): The podcast topic
        docs (List): A list of Wikipedia documents to summarize

    Returns:
        str: A summary string representing the batch of documents
    """
    prompthub_path = "piyushkappa/summarization_chunk"
    summary_prompt = hub.pull(prompthub_path)

    llm = get_fast_llm(config)
    summary_chain = summary_prompt | llm

    context_text = "\n\n".join([format_wikipedia_document(d) for d in docs])
    result = summary_chain.invoke({"topic": topic, "documents": context_text})

    return result if isinstance(result, str) else str(result)


def outline_episode(
    config: PodcastConfig, topic: str, background_info: list, chunk_size: int = 5
) -> PodcastOutline:
    """
    Generate a structured outline for a podcast episode using map–reduce.

    - Map step: Summarize background docs in parallel (chunks).
    - Reduce step: Combine summaries and generate final outline.

    Args:
        topic (str): The main topic for the podcast episode
        background_info (list): List of Wikipedia document objects containing research material
        chunk_size (int): Number of documents per summarization chunk

    Returns:
        PodcastOutline: Structured outline object containing sections and subsections
    """
    logger.info(f"Generating outline for podcast on: {topic}")

    # === Map step: parallel summarization ===
    doc_chunks = [
        background_info[i : i + chunk_size]
        for i in range(0, len(background_info), chunk_size)
    ]
    logger.info(
        f"Splitting {len(background_info)} documents into {len(doc_chunks)} chunks"
    )
    with ThreadPoolExecutor(max_workers=8) as executor:  # tune workers
        summaries = list(
            executor.map(
                lambda chunk: summarize_background_chunk(config, topic, chunk),
                doc_chunks,
            )
        )
    logger.info(f"Got {len(summaries)} summaries from background documents")

    # === Reduce step: final outline generation ===
    prompthub_path = "evandempsey/podcast_outline:6ceaa688"
    outline_prompt = hub.pull(prompthub_path)
    outline_llm = get_long_context_llm(config)

    outline_chain = outline_prompt | outline_llm.with_structured_output(PodcastOutline)

    outline = outline_chain.invoke(
        {
            "episode_structure": config.episode_structure_for_prompt,
            "topic": topic,
            "context_documents": "\n\n".join(summaries),
        }
    )

    logger.info(outline.as_str)
    return outline
