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

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['AUDIO_FOLDER'] = 'static/audio/'
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
    max_depth=2,
    max_links_per_page=3
)
API_URL = os.environ["API_URL"]
TTS_HEADERS = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}

class ChatSession:
    def __init__(self):
        self.messages = []
        self.vector_store = None
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
    print("AAAA create tts")
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

def process_pdf(file_path):
    with fitz.open(file_path) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text()
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    # Create vector store
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    return vector_store, len(chunks)

def process_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return "Image text: " + text

@app.route('/')
def home():
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = ChatSession()
    return render_template('index.html', session_id=session_id)

@app.route('/chat', methods=['POST'])
def chat():
    print(f"Incoming request data: {request.get_json()}")
    data = request.json
    session_id = data.get('session_id')
    message = data.get('message')
    message_type = data.get('message_type', 'text') 
    tts_enabled = data.get('tts_enabled', True)
    
    print("no")
    if not session_id or session_id not in chat_sessions:
        print("fuck")
        return jsonify({'error': 'Invalid session'}), 400
        
    session = chat_sessions[session_id]
    
    # Add user message to history
    print("aa")
    if message_type == 'text':
        print(session)
        session.add_message('user', message)
        print(message)

        # Placeholder/Loading message
        processing_message = session.add_message('assistant', "Thinking...")
        update_chat_async(session_id, session.messages) 
    
        # Generate response based on context
        if session.vector_store:
            # RAG-based response for PDF context
            docs = session.vector_store.similarity_search(message, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            response = agent.generate_rag_response(message, context)
        
        elif session.image_text:
            response = agent.generate_rag_response(message, session.image_text)

        else:
            # Web search-based response
            response = agent.search_and_analyze(message)
    
        # Generate audio for response
        if tts_enabled:
            audio_filename = f"{uuid.uuid4()}.wav"
            audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
            tts_success = create_tts(response, audio_path)
        else:
            tts_success = False

        processing_message['content'] = response
        processing_message['audio_file'] = audio_filename if tts_success else None
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
        vector_store, chunk_count = process_pdf(file_path)
        session.vector_store = vector_store

        
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
            print(text)
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