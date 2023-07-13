# Transcript Summarization App

This flask app provides an interface for summarizing meeting transcripts using pretrained language models.

## Instructions
To run the app locally:

1. Ensure you have Python 3.7+ and Flask installed
2. Clone this repo
3. Run pip install requirements.txt
4. Include the desired model api keys in your environment variables or create a .env file that includes them. See .env.example for an example.
5. Run `python MSA.py`
6. Visit http://127.0.0.1:5000 in your browser

The app accepts .vtt video transcripts for summarization, but can also take raw text input or .txt files. 

## Usage

Create a config.py and supply your OpenAI and Anthropic API keys.

On the home page, you can either upload a .vtt transcript file, enter raw text, or paste the contents of a .txt file. 

In the "Model Selection" dropdown, choose between:

- GPT-3.5: an older version of OpenAI's GPT-3 model
- GPT-4: OpenAI's latest offering
- Claude-2: Anthropic's Constitutional AI model

Click "Summarize" to generate a bulleted summary of the meeting discussion points using the selected model.

You can then download the summary as a .txt file or edit it on the page and re-summarize.

## Overview
This app provides a wrapper around a meeting transcript summarizer. It uses a token splitter to chunk long input texts and employs batch inference to efficiently summarize each chunk with the selected language model.

The summarizer aims to capture the essence of meeting discussions in a concise yet comprehensive bulleted list format.
