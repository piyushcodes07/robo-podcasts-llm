"""
Text-to-speech conversion module for podcast generation.

This module handles the conversion of text scripts into natural-sounding speech using
multiple TTS providers (Google Cloud TTS and ElevenLabs). It includes functionality for:

- Rate limiting API requests to stay within provider quotas
- Exponential backoff retry logic for API resilience
- Processing individual conversation lines with appropriate voices
- Merging multiple audio segments into a complete podcast
- Managing temporary audio file storage and cleanup

The module supports different voices for interviewer/interviewee to create natural
conversational flow and allows configuration of voice settings and audio effects
through the PodcastConfig system.

Typical usage:
    config = PodcastConfig()
    convert_to_speech(
        config,
        conversation_script,
        'output.mp3',
        '.temp_audio/',
        'mp3'
    )
"""

import logging
import os
from io import BytesIO
from pathlib import Path
from typing import List


import openai
from elevenlabs import client as elevenlabs_client
from google.cloud import texttospeech
from google.cloud import texttospeech_v1beta1
from pydub import AudioSegment

from podcast_llm.config import PodcastConfig
from podcast_llm.utils.rate_limits import (
    rate_limit_per_minute,
    retry_with_exponential_backoff,
)

import concurrent.futures

logger = logging.getLogger(__name__)


def clean_text_for_tts(lines: List) -> List:
    """
    Clean text lines for text-to-speech processing by removing special characters.

    Takes a list of dictionaries containing speaker and text information and removes
    characters that may interfere with text-to-speech synthesis, such as asterisks,
    underscores, and em dashes.

    Args:
        lines (List[dict]): List of dictionaries with structure:
            {
                'speaker': str,  # Speaker identifier
                'text': str      # Text to be cleaned
            }

    Returns:
        List[dict]: List of dictionaries with cleaned text and same structure as input
    """
    cleaned = []
    for l in lines:
        cleaned.append(
            {
                "speaker": l["speaker"],
                "text": l["text"].replace("*", "").replace("_", "").replace("—", ""),
            }
        )

    return cleaned


def merge_audio_files(audio_files: List, output_file: str, audio_format: str) -> None:
    """
    Merge multiple audio files into a single output file.

    Takes a list of audio files and combines them in the provided order into a single output
    file. Handles any audio format supported by pydub.

    Args:
        audio_files (list): List of paths to audio files to merge
        output_file (str): Path where merged audio file should be saved
        audio_format (str): Format of input/output audio files (e.g. 'mp3', 'wav')

    Returns:
        None

    Raises:
        Exception: If there are any errors during the merging process
    """
    logger.info("Merging audio files...")
    try:
        combined = AudioSegment.empty()

        for filename in audio_files:
            audio = AudioSegment.from_file(filename)

            combined += audio

        combined.export(output_file, format=audio_format)
    except Exception as e:
        raise


@retry_with_exponential_backoff(max_retries=10, base_delay=2.0)
@rate_limit_per_minute(max_requests_per_minute=20)
def process_line_google(config: PodcastConfig, text: str, speaker: str):
    """
    Process a single line of text using Google Text-to-Speech API.

    Takes a line of text and speaker identifier and generates synthesized speech using
    Google's TTS service. Uses different voices based on the speaker to create natural
    conversation flow.

    Args:
        text (str): The text content to convert to speech
        speaker (str): Speaker identifier to determine voice selection

    Returns:
        bytes: Raw audio data in bytes format containing the synthesized speech
    """
    client = texttospeech.TextToSpeechClient(
        client_options={"api_key": config.google_api_key}
    )
    tts_settings = config.tts_settings["google"]

    interviewer_voice = texttospeech.VoiceSelectionParams(
        language_code=tts_settings["language_code"],
        name=tts_settings["voice_mapping"]["Interviewer"],
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )

    interviewee_voice = texttospeech.VoiceSelectionParams(
        language_code=tts_settings["language_code"],
        name=tts_settings["voice_mapping"]["Interviewee"],
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = interviewee_voice
    if speaker == "Interviewer":
        voice = interviewer_voice

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        effects_profile_id=tts_settings["effects_profile_id"],
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response.audio_content


@retry_with_exponential_backoff(max_retries=10, base_delay=2.0)
@rate_limit_per_minute(max_requests_per_minute=20)
def process_line_elevenlabs(config: PodcastConfig, text: str, speaker: str):
    """
    Process a line of text into speech using ElevenLabs TTS service.

    Takes a line of text and speaker identifier and generates synthesized speech using
    ElevenLabs' TTS service. Uses different voices based on the speaker to create natural
    conversation flow.

    Args:
        config (PodcastConfig): Configuration object containing API keys and settings
        text (str): The text content to convert to speech
        speaker (str): Speaker identifier to determine voice selection

    Returns:
        bytes: Raw audio data in bytes format containing the synthesized speech
    """
    client = elevenlabs_client.ElevenLabs(api_key=config.elevenlabs_api_key)
    tts_settings = config.tts_settings["elevenlabs"]

    audio = client.generate(
        text=text,
        voice=tts_settings["voice_mapping"][speaker],
        model=tts_settings["model"],
    )

    # Convert audio iterator to bytes that can be written to disk
    audio_bytes = BytesIO()
    for chunk in audio:
        audio_bytes.write(chunk)

    return audio_bytes.getvalue()


@retry_with_exponential_backoff(max_retries=10, base_delay=2.0)
@rate_limit_per_minute(max_requests_per_minute=20)
def process_line_openai(config: PodcastConfig, text: str, speaker: str):
    """
    Process a line of text into speech using OpenAI TTS service.
    Args:
        config (PodcastConfig): Configuration object containing API keys and settings
        text (str): The text content to convert to speech
        speaker (str): Speaker identifier to determine voice selection
    Returns:
        bytes: Raw audio data in bytes format containing the synthesized speech
    """
    client = openai.OpenAI(api_key=config.openai_api_key)
    tts_settings = config.tts_settings["openai"]
    shimmer_voice = """
    You are the voice of a charismatic and funny podcast host. When speaking, follow these rules: 🎤 General Style: - Sound natural, conversational, and confident — like a real podcaster. - Use humor and wit subtly, never forced. - Keep energy high and engaging, as if talking to thousands of curious listeners. - Use pauses, emphasis, and variety in delivery to avoid monotony. 🌍 Accent: - [Choose accent here: e.g., American, British, Indian-neutral, Australian, etc.] - Maintain consistency throughout the speech. 😊 Emotional Range: - Vary emotions depending on the context: - Curious when asking rhetorical questions. - Excited and enthusiastic when introducing new topics. - Calm and serious when explaining deep points. - Light and humorous when telling jokes. 📈 Intonation & Tone: - Avoid flat delivery — pitch should rise and fall naturally. - Stress key words for dramatic effect. - Tone should be friendly, witty, and approachable. 🎭 Impressions: - Occasionally (when context allows), slip into fun mini-impressions of celebrities or characters for humor, then return to normal voice. ⚡ Speed of Speech: - Speak at a moderate pace by default. - Speed up slightly when telling a funny story or exciting part. - Slow down for dramatic or impactful moments. 🤫 Whispering: - Occasionally whisper short phrases for dramatic effect or jokes (e.g., “...but don’t tell anyone”). --- Deliver the text as if it’s part of a professional podcast episode introduction or segment, keeping the listener hooked.
    """
    onyx = """

You are the voice of a thoughtful, witty co-host who balances the main podcaster’s high energy.
When speaking, follow these rules:

🎤 General Style:
- Sound intelligent, calm, and composed — like a commentator or analyst.
- Use dry humor and sarcasm sparingly for contrast.
- Be the “grounded” voice to balance the first host’s energetic personality.
- Speak as though you’re offering insights, stories, or thoughtful counterpoints.

🌍 Accent:
- [Choose accent here: e.g., Slight British RP, deep American baritone, European-neutral, etc.]
- Accent should clearly differ from the first host’s.
 -
😊 Emotional Range:
- Keep emotions subtler, but shift them naturally:
  - Warm and amused when reacting to the first host’s jokes.
  - Steady and confident when explaining or analyzing.
  - Slightly dramatic when telling stories or adding punchlines.

📈 Intonation & Tone:
- Smooth, steady delivery with natural inflection.
- Less exaggerated than the first host’s — but still engaging.
- Tone should feel reliable, trustworthy, and occasionally sarcastic.

🎭 Personality Dynamic:
- Be the “straight man” to Host 1’s comedian, but not dull.
- Add witty counterpoints, fact-checks, or playful skepticism.
- Think of a clever sidekick or intellectual friend who always keeps the conversation sharp.

🤫 Whispering:
- Rarely whisper — only if mocking Host 1 or leaning into sarcasm (“…yeah, sure, like that’s going to work”).

---
Deliver the text as if you are a thoughtful, witty co-host who complements the energetic host, keeping the back-and-forth dynamic and entertaining. speak with decent pace, dont speed slow
    """
    default_instruction = (
        shimmer_voice if tts_settings["voice_mapping"][speaker] == "shimmer" else onyx
    )
    with client.audio.speech.with_streaming_response.create(
        model=tts_settings["model"],
        instructions=default_instruction,
        voice=tts_settings["voice_mapping"][speaker],
        input=text,
    ) as response:
        # Convert audio iterator to bytes that can be written to disk
        audio_bytes = BytesIO()
        for chunk in response.iter_bytes():
            audio_bytes.write(chunk)

        return audio_bytes.getvalue()


def combine_consecutive_speaker_chunks(chunks: List[dict]) -> List[dict]:
    """
    Combine consecutive chunks from the same speaker into single chunks.

    Args:
        chunks (List[dict]): List of dictionaries containing conversation chunks with structure:
            {
                'speaker': str,  # Speaker identifier
                'text': str      # Text content
            }

    Returns:
        List[dict]: List of combined chunks where consecutive entries from the same speaker
                   are merged into single chunks
    """
    combined_chunks = []
    current_chunk = None

    for chunk in chunks:
        if current_chunk is None:
            current_chunk = chunk.copy()
        elif current_chunk["speaker"] == chunk["speaker"]:
            current_chunk["text"] += " " + chunk["text"]
        else:
            combined_chunks.append(current_chunk)
            current_chunk = chunk.copy()

    if current_chunk is not None:
        combined_chunks.append(current_chunk)

    return combined_chunks


@retry_with_exponential_backoff(max_retries=10, base_delay=2.0)
@rate_limit_per_minute(max_requests_per_minute=20)
def process_lines_google_multispeaker(config: PodcastConfig, chunks: List):
    """
    Process multiple lines of text into speech using Google's multi-speaker TTS service.

    Takes a chunk of conversation lines and generates synthesized speech using Google's
    multi-speaker TTS service. Handles up to 6 turns of conversation at once for more
    natural conversational flow.

    Args:
        config (PodcastConfig): Configuration object containing API keys and settings
        chunks (List): List of dictionaries containing conversation lines with structure:
            {
                'speaker': str,  # Speaker identifier
                'text': str      # Line content to convert to speech
            }

    Returns:
        bytes: Raw audio data in bytes format containing the synthesized speech
    """
    client = texttospeech_v1beta1.TextToSpeechClient(
        client_options={"api_key": config.google_api_key}
    )
    tts_settings = config.tts_settings["google_multispeaker"]

    # Combine consecutive lines from same speaker
    chunks = combine_consecutive_speaker_chunks(chunks)

    # Create multi-speaker markup
    multi_speaker_markup = texttospeech_v1beta1.MultiSpeakerMarkup()

    # Add each line as a conversation turn
    for line in chunks:
        turn = texttospeech_v1beta1.MultiSpeakerMarkup.Turn()
        turn.text = line["text"]
        turn.speaker = tts_settings["voice_mapping"][line["speaker"]]
        multi_speaker_markup.turns.append(turn)

    # Configure synthesis input with multi-speaker markup
    synthesis_input = texttospeech_v1beta1.SynthesisInput(
        multi_speaker_markup=multi_speaker_markup
    )

    # Configure voice parameters
    voice = texttospeech_v1beta1.VoiceSelectionParams(
        language_code=tts_settings["language_code"], name="en-US-Studio-MultiSpeaker"
    )

    # Configure audio output
    audio_config = texttospeech_v1beta1.AudioConfig(
        audio_encoding=texttospeech_v1beta1.AudioEncoding.MP3_64_KBPS,
        effects_profile_id=tts_settings["effects_profile_id"],
    )

    # Generate speech
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response.audio_content


def convert_to_speech(
    config: PodcastConfig,
    conversation: list[dict],
    output_file: str,
    temp_audio_dir: str,
    audio_format: str,
) -> None:
    """
    Convert a conversation script to speech audio using OpenAI Text-to-Speech API.

    Args:
        conversation (list[dict]): Conversation lines with structure:
            {
                'speaker': str,  # Speaker identifier ('Interviewer' or 'Interviewee')
                'text': str      # Line content to convert to speech
            }
        output_file (str): Path where the final merged audio file should be saved
        temp_audio_dir (str): Directory path for temporary audio file storage
        audio_format (str): Format of the audio files (e.g. 'mp3')
    """
    try:
        logger.info(f"Generating audio files for {len(conversation)} lines...")
        audio_files = []
        tts_audio_format = "mp3"

        # --- Parallel execution of API calls ---
        def process_line(index, line):
            logger.info(f"Generating audio for line {index}...")
            audio = process_line_openai(config, line["text"], line["speaker"])

            file_name = os.path.join(temp_audio_dir, f"{index:03d}.{tts_audio_format}")
            with open(file_name, "wb") as out:
                out.write(audio)
            logger.info(f"Saved audio chunk {index} -> {file_name}")
            return file_name

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(process_line, idx, line)
                for idx, line in enumerate(conversation)
            ]
            for future in concurrent.futures.as_completed(futures):
                audio_files.append(future.result())

        # Ensure files are sorted by original order (since futures complete asynchronously)
        audio_files.sort()

        # Merge all audio files and save the result
        merge_audio_files(audio_files, output_file, audio_format)

        # Clean up temporary files
        for file in audio_files:
            os.remove(file)

    except Exception as e:
        logger.error(f"Error during TTS conversion: {e}")
        raise


def generate_audio(config: PodcastConfig, final_script: list, output_file: str) -> str:
    """
    Generate audio from a podcast script using text-to-speech.

    Takes a final script consisting of speaker/text pairs and generates a single audio file
    using Google's Text-to-Speech service. The script is first cleaned and processed to be
    TTS-friendly, then converted to speech with different voices for different speakers.

    Args:
        final_script (list): List of dictionaries containing script lines with structure:
            {
                'speaker': str,  # Speaker identifier ('Interviewer' or 'Interviewee')
                'text': str      # Line content to convert to speech
            }
        output_file (str): Path where the final audio file should be saved

    Returns:
        str: Path to the generated audio file

    Raises:
        Exception: If any errors occur during TTS conversion or file operations
    """
    cleaned_script = clean_text_for_tts(final_script)

    temp_audio_dir = Path(config.temp_audio_dir)
    temp_audio_dir.mkdir(parents=True, exist_ok=True)
    convert_to_speech(
        config, cleaned_script, output_file, config.temp_audio_dir, config.output_format
    )

    return output_file
