import os
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from podcast_llm.generate import generate, parse_arguments, DEFAULT_CONFIG_PATH
from podcast_llm.config import PodcastConfig
from podcast_llm.models import (
    PodcastOutline,
    PodcastSection,
    PodcastSubsection
)


# Simplified mock_checkpointer for testing
def mock_checkpointer_instance_for_test():
    mock = AsyncMock()
    mock.checkpoint_async.side_effect = [
        ['background'], # For research_background_info
        PodcastOutline(sections=[PodcastSection(title='Section1', subsections=[PodcastSubsection(title='Subsection 1')])]), # For outline_episode
        ['topics'], # For research_discussion_topics
        ['draft'], # For write_draft_script
        [{'speaker': 'Interviewer', 'text': 'Hello'}] # For write_final_script
    ]
    return mock


@pytest.mark.asyncio
@patch('podcast_llm.generate.research_background_info', new_callable=AsyncMock)
@patch('podcast_llm.generate.outline_episode', new_callable=Mock) # outline_episode is synchronous
@patch('podcast_llm.generate.research_discussion_topics', new_callable=AsyncMock)
@patch('podcast_llm.generate.write_draft_script', new_callable=AsyncMock)
@patch('podcast_llm.generate.write_final_script', new_callable=AsyncMock)
@patch('podcast_llm.generate.generate_audio', new_callable=AsyncMock)
@patch('podcast_llm.generate.Checkpointer')
async def test_generate_with_audio_and_text_output(
    mock_checkpointer_class,
    mock_generate_audio,
    mock_write_final,
    mock_write_draft,
    mock_research_topics,
    mock_outline,
    mock_research_background,
    tmp_path: Path
) -> None:
    """Test full podcast generation with both audio and text output."""
    # Setup
    mock_checkpointer_class.return_value = mock_checkpointer_instance_for_test() # Use simplified mock
    mock_research_background.return_value = ['background']
    mock_outline.return_value = PodcastOutline(sections=[
        PodcastSection(title='Section1', subsections=[
            PodcastSubsection(title='Subsection 1'), PodcastSubsection(title='Subsection 2')
        ])
    ])
    mock_research_topics.return_value = ['topics']
    mock_write_draft.return_value = ['draft']
    mock_write_final.return_value = [{'speaker': 'Interviewer', 'text': 'Hello'}]
    
    audio_output = tmp_path / 'test.mp3'
    text_output = tmp_path / 'test.md'

    for e in ['GOOGLE_API_KEY', 'ELEVENLABS_API_KEY',  'OPENAI_API_KEY',
              'TAVILY_API_KEY', 'ANTHROPIC_API_KEY']:
        os.environ[e] = 'foo'

    # Execute
    await generate(
        topic='test topic',
        mode='research',
        qa_rounds=2,
        use_checkpoints=True,
        audio_output=str(audio_output),
        text_output=str(text_output),
        config=DEFAULT_CONFIG_PATH,
        debug=True
    )

    # Verify
    mock_generate_audio.assert_called_once()
    assert text_output.exists()


@pytest.mark.asyncio
async def test_generate_without_outputs() -> None:
    """Test generation without audio or text output."""
    with patch('podcast_llm.generate.Checkpointer') as mock_checkpointer_class:
        mock_checkpointer_instance = mock_checkpointer_instance_for_test()
        mock_checkpointer_class.return_value = mock_checkpointer_instance

        for e in ['GOOGLE_API_KEY', 'ELEVENLABS_API_KEY',  'OPENAI_API_KEY',
                  'TAVILY_API_KEY', 'ANTHROPIC_API_KEY']:
            os.environ[e] = 'foo'
        
        await generate(
            topic='test topic',
            mode='research',
            qa_rounds=2,
            use_checkpoints=True,
            audio_output=None,
            text_output=None,
            config=DEFAULT_CONFIG_PATH,
            debug=False
        )

        mock_checkpointer_instance.checkpoint_async.assert_called()


def test_parse_arguments() -> None:
    """Test command line argument parsing."""
    test_args = ['test topic', '--qa-rounds', '3', '--debug']
    with patch('sys.argv', ['script.py'] + test_args):
        args = parse_arguments()
        
        assert args.topic == 'test topic'
        assert args.qa_rounds == 3
        assert args.debug is True
        assert args.checkpoint is True
        assert args.audio_output is None
        assert args.text_output is None


def test_parse_arguments_with_outputs() -> None:
    """Test parsing arguments with output paths specified."""
    test_args = [
        'test topic',
        '--audio-output', 'test.mp3',
        '--text-output', 'test.md',
        '--no-checkpoint'
    ]
    
    with patch('sys.argv', ['script.py'] + test_args):
        args = parse_arguments()
        
        assert args.audio_output == 'test.mp3'
        assert args.text_output == 'test.md'
        assert args.checkpoint is False