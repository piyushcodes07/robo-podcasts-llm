import pytest
from podcast_llm.extractors.youtube import YouTubeSourceDocument
from unittest.mock import AsyncMock


@pytest.mark.parametrize('url,expected_id', [
    ('https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'dQw4w9WgXcQ'),
    ('https://youtu.be/dQw4w9WgXcQ', 'dQw4w9WgXcQ'),
    ('https://www.youtube.com/embed/dQw4w9WgXcQ', 'dQw4w9WgXcQ'),
    ('dQw4w9WgXcQ', 'dQw4w9WgXcQ'),
    ('https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=123', 'dQw4w9WgXcQ'),
    ('https://youtu.be/dQw4w9WgXcQ?t=123', 'dQw4w9WgXcQ'),
    ('https://www.youtube.com/shorts/f7ZNtQZPha8', 'f7ZNtQZPha8'),
    ('https://youtube.com/shorts/HJrbhrsODMk?si=XNDlfvA9JfgbM_WR', 'HJrbhrsODMk'),
])
def test_extract_video_id(url: str, expected_id: str) -> None:
    """Test that video IDs are correctly extracted from various URL formats."""
    extractor = YouTubeSourceDocument(url)
    assert extractor.video_id == expected_id


@pytest.mark.asyncio
async def test_extract_transcript(mocker) -> None:
    """Test transcript extraction with mocked YouTube API."""
    mock_transcript = [
        {'text': 'First line'},
        {'text': 'Second line'},
        {'text': 'Third line'}
    ]
    mocker.patch(
        'podcast_llm.extractors.youtube.YouTubeTranscriptApi.get_transcript',
        return_value=mock_transcript
    )
    
    extractor = YouTubeSourceDocument('test_video_id')
    transcript = await extractor.extract()
    
    assert transcript == 'First line Second line Third line'
    assert extractor.content == transcript


@pytest.mark.asyncio
async def test_extract_transcript_failure(mocker) -> None:
    """Test transcript extraction handles API failures gracefully."""
    mocker.patch(
        'podcast_llm.extractors.youtube.YouTubeTranscriptApi.get_transcript',
        side_effect=Exception('API Error')
    )
    
    extractor = YouTubeSourceDocument('test_video_id')
    with pytest.raises(Exception):
        await extractor.extract()