from flask import Flask, render_template, request, Response, make_response, session, redirect, url_for
import os


from pathlib import Path
from typing import List, Any
from expert import LanguageExpert
from langchain.text_splitter import TokenTextSplitter
import webvtt

global bullet_expert
bullet_expert = {'name': 'Meetings Minutes Writer', 
                 'system_message': 'You are a helpful assitant that generates bulleted meeting summaries.',  
                 'model_params': {'model_name': 'claude-2', 'temperature': 0.7, 'frequency_penalty': 1.0, 'presence_penalty': 0.5, 'n': 1, 'max_tokens': 2048}}


def chunk_text(text: str, chunk_size: int = 2048, chunk_overlap: int = 0) -> List[str]:
    """Split the text into chunks of a specified size."""
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def batch_list(items: List[Any], batch_size: int = 10) -> List[List[Any]]:
    """Split a list into smaller lists of a specified size."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def convert_vtt_to_txt(infile, outfile):
    """Convert VTT subtitle file to a plain text file."""
    vtt = webvtt.read(infile)
    transcript = ""
    lines = []
    last_speaker = None
    for line in vtt:
        speaker = line.lines[0].split('>')[0].split('v ')[1]
        if last_speaker != speaker:
            lines.append('\n'+speaker + ': ')
        lines.extend(line.text.strip().splitlines())
        last_speaker = speaker
    previous = None
    for line in lines:
        if line == previous:
            continue
        transcript += f" {line}"
        previous = line
    with open(outfile, 'w') as f:
        f.write(transcript)
    print(f'Length of original:\t{len(vtt.content)} characters\nLength of final:\t{len(transcript)} characters\nPercent Reduction:\t{100 - len(transcript)*100/len(vtt.content):.0f}%')



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# Ensure that the Flask app has a secret key to use sessions
app.secret_key = os.urandom(24)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method != 'POST':
        return render_template('index.html')
    # Check if file is uploaded
    is_file_uploaded = 'file' in request.files and request.files['file'].filename != ''
    # Check if text is entered
    is_text_entered = 'text' in request.form and request.form['text'].strip() != ''

    if not is_file_uploaded and not is_text_entered:
        return 'No file or text provided', 400

    if is_file_uploaded:
        file = request.files['file']
        input_file = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(input_file)
        if input_file.endswith('.vtt'):
            convert_vtt_to_txt(input_file, f'{input_file[:-4]}.txt')
            input_file = f'{input_file[:-4]}.txt'
        text_content = Path(input_file).read_text(encoding='utf-8')
    else:
        text_content = request.form['text']

    # (Perform the summarization code, as before...)
    selected_model = request.form['model_name']
    summary = summarize(text_content, selected_model)
    session['summary'] = summary
    # Instead of creating a downloadable file, render the template with the summary
    return redirect(url_for('display_summary'))

@app.route('/download-summary', methods=['GET', 'POST'])
def download_summary():
    if request.method == 'POST':
        # Extract the edited summary from the request if available
        summary = request.form.get('edited-summary', '')
    else:
        # Get the summary from the user session for GET requests
        summary = session.get('summary', '')

    # Prepare the response as a downloadable text file
    response = make_response(summary)
    response.headers.set('Content-Type', 'text/plain')
    response.headers.set('Content-Disposition', 'attachment', filename='summary.txt')

    return response

@app.route('/summary', methods=['GET', 'POST'])
def display_summary():
    # Get the summary from the user session
    summary = session.get('summary', '')

    if request.method == 'POST':
        if edited_summary := request.form.get('edited-summary', ''):
            summary = edited_summary
            session['summary'] = summary

    # Render the template with the summary
    return render_template('index.html', summary=summary)


def summarize(text_content, selected_model):
    max_tokens = 75000 if 'claude' in selected_model else 4096
    chunks_of_text_content: List[str] = chunk_text(text_content, chunk_size=max_tokens)
    chunks_of_text_content: List[str] = [f'Summarize the following meeting in a bulleted format. Include as many details as possible. The summary should be about 1000 words long.\n\n<transcript>{chunk}</transcript>' for chunk in chunks_of_text_content]
    batched_chunks: List[List[str]] = batch_list(chunks_of_text_content)

    bullet_generator: LanguageExpert = LanguageExpert(**bullet_expert)
    bullet_generator.change_param("model_name", selected_model)

    summarized_chunks: List[str] = []
    for batch in batched_chunks:
        summarized_batch: List[str] = bullet_generator.bulk_generate(batch)
        summarized_chunks.extend(summarized_batch)

    return ''.join(summarized_chunks)

if __name__ == '__main__':
    app.run(debug=True) 