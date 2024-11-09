# app.py
from flask import Flask, render_template, request, jsonify, send_file
import os
from dotenv import load_dotenv
import threading
import uuid
from datetime import datetime
import fitz
from PIL import Image
import pytesseract
import requests
from werkzeug.utils import secure_filename
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import numpy as np
from web_search_agent import WebSearchAgent
import io
from document_Manager import DocumentManager
from select_best_sources import SelectBestSources
from LaIA_dialogue import LaIA_dialogue
from LaIA_video import LaIA_video
import re
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['AUDIO_FOLDER'] = 'static/audio/'
app.config['VIDEO_FOLDER'] = 'static/video/'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB limit

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

agent = WebSearchAgent(
    base_url=os.environ["BASE_URL"],
    api_key=os.environ["HF_TOKEN"],
    max_depth=1,
    max_links_per_page=1
)
API_URL = os.environ["API_URL"]
TTS_HEADERS = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}

class ChatSession:
    def __init__(self):
        self.messages = []
        self.document_manager = None
        self.image_text = None
        
    def add_message(self, role, content, audio_file=None):
        message = {
            'id': str(uuid.uuid4()),
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'audio_file': audio_file
        }
        self.messages.append(message)
        return message

chat_sessions = {}

def create_tts(text, filename):
    API_URL = os.environ["API_URL"]
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    
    response = requests.post(API_URL, headers=headers, json={"text": text, "voice": 20})
    print(response)
    
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        return True
    print("Oh no")
    return False

def process_pdf_text(file_path):
    with fitz.open(file_path) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text()
    
    return text

def process_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return "Image text: " + text

def detect_video_request(text):
    # Regular expression to detect keywords related to video or audio content
    text = text.lower()
    pattern = r'\b(vídeo|video|tiktok|reels|podcast)\b'
    
    # Search for the pattern in the text, case insensitive
    return bool(re.search(pattern, text, re.IGNORECASE))

@app.route('/')
def home():
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = ChatSession()
    chat_sessions[session_id].document_manager = DocumentManager(embeddings, text_splitter)

    return render_template('index.html', session_id=session_id)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
        
    audio_file = request.files['audio']
    session_id = request.form.get('session_id')
    
    if not session_id or session_id not in chat_sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    try:
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            audio_file.save(tmp_file.name)
            
            # Start recording and get transcription
            transcription = stt_service.start_recording(tmp_file.name)
            
            # Clean up the temporary file
            os.unlink(tmp_file.name)
            
            return jsonify({
                'transcription': transcription,
                'success': True
            })
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/chat', methods=['POST'])
def chat():
    print(f"Incoming request data: {request.get_json()}")
    data = request.json
    session_id = data.get('session_id')
    message = data.get('message')
    message_type = data.get('message_type', 'text') 
    tts_enabled = data.get('tts_enabled', True)
    
    if not session_id or session_id not in chat_sessions:
        return jsonify({'error': 'Invalid session'}), 400
        
    session = chat_sessions[session_id]
    
    # Add user message to history
    if message_type == 'text':
        print(session)
        session.add_message('user', message)
        print(message)

        # Placeholder/Loading message
        processing_message = session.add_message('assistant', "Thinking...")
        update_chat_async(session_id, session.messages) 


        if detect_video_request(message):
            ubi = app.config['VIDEO_FOLDER'] + "final_video_with_subtitles.mp4"
            socketio.emit('video_generation_start', {
            }, room=session_id)


            
            text = session.document_manager.get_context(query=message, llm_client=agent.client)
            print(text)
            socketio.emit('video_progress', {
                'progress': 20,
                'status': 'Generating dialogue...'
            }, room=session_id)

            print("aaa")
            
            dialog = LaIA_dialogue(text)
            dialog.create_dialogue()
            
            socketio.emit('video_progress', {
                'progress': 40,
                'status': 'Creating video scenes...'
            }, room=session_id)

            print(dialog.dialogue)
            
            video = LaIA_video(dialog.dialogue, final_video_ubi=ubi)
            
            socketio.emit('video_progress', {
                'progress': 80,
                'status': 'Adding subtitles...'
            }, room=session_id)
            
            # ... video processing ...
            
            socketio.emit('video_progress', {
                'progress': 100,
                'status': 'Finalizing video...'
            }, room=session_id)
            
            socketio.emit('video_generation_complete', room=session_id)
            
            processing_message['content'] = "Aquí tens el teu video."
            processing_message['citations'] = []
            processing_message['audio_file'] = None
            processing_message['video_file'] = ubi
            return jsonify({'history': session.messages})

    
        # Generate response based on context
        if len(session.document_manager.documents) != 0:
            # RAG-based response for PDF context
            print("ragresponse")
            
            output = session.document_manager.generate_response(
                query=message,
                llm_client=agent.client,
                include_citations=True
            )

            response = output['response']
            citations = output['citations']

            if response == 'No relevant context found to answer the question.':
                query = agent.process_prompt(message)
                base_url = os.environ["BASE_URL"]
                api_key = os.environ["HF_TOKEN"]
                select_best_sources = SelectBestSources(base_url=base_url, api_key=api_key, max_source_chars_length=500, max_simultaneous_sources=5, remove_parent_urls=False)
                query = query.split("\n")
                for q in query[:3]:
                # Get response
                    output = agent.search_and_analyze(q)
                    if output != False:
                        select_best_sources.append_sources(output)
                responses = select_best_sources.get_final_sources(message)
                for r in responses:
                    session.document_manager.add_document(
                        title=f"Web Search: {r[0]}",
                        content=r[1],
                        doc_type='web',
                        source_url=r[0]
                    )
                output = session.document_manager.generate_response(
                    query=message,
                    llm_client=agent.client,
                    include_citations=True
                )
                response = output['response']
                citations = output['citations']

        else:
            # Web search-based response

            query = agent.process_prompt(message)
            base_url = os.environ["BASE_URL"]
            api_key = os.environ["HF_TOKEN"]
            select_best_sources = SelectBestSources(base_url=base_url, api_key=api_key, max_source_chars_length=500, max_simultaneous_sources=5, remove_parent_urls=False)
            query = query.split("\n")
            for q in query[:3]:
            # Get response
                output = agent.search_and_analyze(q)
                if output != False:
                    select_best_sources.append_sources(output)
            responses = select_best_sources.get_final_sources(message)
            for r in responses:
                session.document_manager.add_document(
                    title=f"Web Search: {r[0]}",
                    content=r[1],
                    doc_type='web',
                    source_url=r[0]
                )
            output = session.document_manager.generate_response(
                query=message,
                llm_client=agent.client,
                include_citations=True
            )
            response = output['response']
            citations = output['citations']
    
        # Generate audio for response
        if tts_enabled:
            audio_filename = f"{uuid.uuid4()}.wav"
            audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
            tts_success = create_tts(response, audio_path)
        else:
            tts_success = False

        processing_message['content'] = response
        processing_message['citations'] = citations
        processing_message['audio_file'] = audio_filename if tts_success else None
        processing_message['video_file'] = None
        return jsonify({'history': session.messages})
    
    elif message_type == 'audio':
        pass

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    
    session_id = request.form.get('session_id')

    
    if not session_id or session_id not in chat_sessions:
        return jsonify({'error': 'Invalid session'}), 400
        
    session = chat_sessions[session_id]
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process PDF and create vector store
        text = process_pdf_text(file_path)

        document, chunk_count = session.document_manager.add_document(
            title=file.filename,
            content=text,
            doc_type='pdf'
        )

        
        # Add system message about PDF processing
        system_msg = f"PDF '{filename}' processed successfully. Created {chunk_count} chunks for analysis. You can now ask questions about the document."
        message = session.add_message('system', system_msg)
        
        return jsonify({
            'message': message,
            'history': session.messages
        })
    
    elif file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        try:
            text = process_image(file)
            session.image_text = text  # Store extracted text

            document, chunk_count = session.document_manager.add_document(
                title=file.filename,
                content=text,
                doc_type='image'
            )

            message = session.add_message('system', f"Image processed. Extracted text: {text[:100]}...")
        except Exception as e:
            message = session.add_message('system', f"Error processing image: {e}")
        return jsonify({'message': message, 'history': session.messages})

    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/clear-context', methods=['POST'])
def clear_context():
    session_id = request.form.get('session_id')
    if session_id and session_id in chat_sessions:
        session = chat_sessions[session_id]
        session.vector_store = None  # Clear PDF context
        session.image_text = None   # Clear image text
        message = session.add_message('system', "Context cleared. Now using web search.")
        return jsonify({'message': message, 'history': session.messages})
    return jsonify({'error': 'Invalid session'}), 400

def update_chat_async(session_id, messages):
    with app.app_context():  # Important: Access Flask context
        data = {'history': messages}
        socketio.emit('chat_update', data, room=session_id, namespace='/chat')


@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_file(
        os.path.join(app.config['AUDIO_FOLDER'], filename),
        mimetype='audio/wav'
    )

@app.route('/documents', methods=['POST'])
def get_documents():
    session_id = request.json.get('session_id')  # Use request.json for JSON data
    if session_id and session_id in chat_sessions:
        session = chat_sessions[session_id]
        return jsonify({'documents': session.document_manager.get_document_list()})
    return jsonify({'error': 'Invalid session ID'}), 400



@app.route('/url', methods=['POST'])
def push_url():
    print("RECIEVED SOMETHING")
    session_id = request.json.get('session_id')  # Use request.json for JSON data
    url = request.json.get('url')
    print("Here's the url", url)
    if session_id and session_id in chat_sessions:
        newweb = agent.simple_web(url)
        session = chat_sessions[session_id]
        document, chunk_count = session.document_manager.add_document(
            title=f"Web Search: {url}",
            content=newweb,
            doc_type='web',
            source_url = url
        )

        return jsonify({'text': "Done"})
    return jsonify({'error': 'Invalid session ID'}), 400



@app.route('/documents/<doc_id>', methods=['DELETE'])
def remove_document(doc_id):
    session_id = request.form.get('session_id')
    if session_id and session_id in chat_sessions:
        session = chat_sessions[session_id]
        success = session.document_manager.remove_document(doc_id)
        return jsonify({
            'success': success,
            'documents': session.document_manager.get_document_list()
        })
    return jsonify({'error': 'Invalid file type'}), 400

from flask_socketio import SocketIO, join_room, leave_room

socketio = SocketIO(app, async_mode='threading') # Important: use threading for async


@socketio.on('connect', namespace='/chat')
def handle_connect():
    session_id = request.args.get('session_id')
    if session_id:
        join_room(session_id, namespace='/chat') # Use namespace here
@socketio.on('disconnect', namespace='/chat')
def handle_disconnect():
    session_id = request.args.get('session_id')
    if session_id:
        leave_room(session_id)

if __name__ == '__main__':
    socketio.run(app, debug=True)