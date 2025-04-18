import streamlit as st
import librosa
import numpy as np
import io
import random

# Set page configuration
st.set_page_config(
    page_title="Grammar Scoring Engine",
    page_icon="🎙️",
    layout="centered"
)

# App title and description
st.title("🎙️ Grammar Scoring Engine")
st.markdown("Upload your voice recording to analyze your spoken grammar.")

# =======================
# 1. Upload and Process Audio
# =======================

def process_audio(audio_file):
    """
    Process uploaded audio file and prepare it for transcription.
    Supports WAV, MP3, and FLAC formats.
    """
    try:
        audio_data, sample_rate = librosa.load(audio_file, sr=None)
        
        # Normalize audio
        normalized_audio = librosa.util.normalize(audio_data)
        
        # Basic noise reduction (optional)
        if len(normalized_audio) > 0:
            noise_level = np.mean(np.abs(normalized_audio[:int(sample_rate/2)]))
            if noise_level > 0.01:
                normalized_audio = normalized_audio - noise_level
                normalized_audio = librosa.util.normalize(normalized_audio)
        
        return normalized_audio, sample_rate, None
    except Exception as e:
        return None, None, str(e)

# =======================
# 2. Placeholder: Transcribe Audio
# =======================

def transcribe_audio(audio_data, sample_rate):
    """
    Placeholder for Whisper transcription. Replace with actual model call.
    """
    # In a real app, you would integrate with a speech-to-text API like Whisper
    # Example: https://github.com/openai/whisper
    
    # For demonstration, we'll return a sample text
    # Sample texts of varying quality
    sample_texts = [
        "This is a perfect transcription with excellent grammar and structure.",
        "This is like a OK transcription with few error in grammar structure.",
        "me talk about the thing that happened yesterday it was very good",
        "when i go to the store i seen many product but couldn't deciding what to buy"
    ]
    
    return {"text": random.choice(sample_texts)}

# =======================
# 3. Placeholder: Analyze Grammar
# =======================

def analyze_with_languagetool(transcription):
    """
    Simulated grammar errors list.
    Replace with actual LanguageTool analysis.
    """
    # In a real app, you would use the LanguageTool API
    # Example: https://languagetool.org/http-api/
    
    # Simulate errors based on text length and complexity
    words = transcription.split()
    num_errors = max(0, int(len(words) * 0.1) + random.randint(-2, 2))
    
    errors = []
    error_types = [
        "Incorrect verb tense", 
        "Subject-verb agreement", 
        "Missing article",
        "Wrong preposition",
        "Spelling error",
        "Punctuation error"
    ]
    
    for _ in range(num_errors):
        errors.append({"error": random.choice(error_types)})
    
    return errors

def analyze_with_spacy(whisper_result):
    """
    Simulated Spacy-like analysis.
    Replace with actual NLP model output.
    """
    # In a real app, you would use spaCy or another NLP tool
    # Example: https://spacy.io/
    
    text = whisper_result["text"]
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    
    if not sentences:
        return {
            "avg_words_per_sentence": 0,
            "complexity_ratio": 0
        }
    
    # Calculate average words per sentence
    words_per_sentence = [len(s.split()) for s in sentences]
    avg_words = sum(words_per_sentence) / len(sentences)
    
    # Calculate lexical complexity (unique words ratio)
    all_words = text.lower().split()
    unique_words = set(all_words)
    complexity = len(unique_words) / max(1, len(all_words))
    
    return {
        "avg_words_per_sentence": avg_words,
        "complexity_ratio": complexity,
    }

# =======================
# 4. Score Calculation
# =======================

def calculate_score(text, language_tool_errors, spacy_analysis):
    """
    Calculate a grammar score based on various factors.
    """
    score = 100

    error_penalty = min(50, len(language_tool_errors) * 3)

    avg_sentence_length = spacy_analysis["avg_words_per_sentence"]
    if avg_sentence_length < 5:
        structure_penalty = 5 * (5 - avg_sentence_length)
    elif avg_sentence_length > 30:
        structure_penalty = 2 * (avg_sentence_length - 30)
    else:
        structure_penalty = 0
    structure_penalty = min(20, structure_penalty)

    complexity_bonus = 0
    if 0.1 <= spacy_analysis["complexity_ratio"] <= 0.4:
        complexity_bonus = 10 * spacy_analysis["complexity_ratio"]

    final_score = max(0, min(100, score - error_penalty - structure_penalty + complexity_bonus))
    return round(final_score, 2)

# =======================
# 5. Generate Feedback
# =======================

def generate_feedback(transcription, language_tool_errors, spacy_analysis):
    feedback = []
    
    if language_tool_errors:
        feedback.append(f"⚠️ Found {len(language_tool_errors)} grammatical issue(s).")
        for i, error in enumerate(language_tool_errors[:5], 1):
            feedback.append(f"  {i}. {error['error']}")
        
        if len(language_tool_errors) > 5:
            feedback.append(f"  ...and {len(language_tool_errors) - 5} more issues.")
    
    if spacy_analysis["avg_words_per_sentence"] < 7:
        feedback.append("📝 Try using slightly longer and more descriptive sentences.")
    
    if spacy_analysis["complexity_ratio"] < 0.25:
        feedback.append("📚 Consider using more varied or complex vocabulary.")
    
    return "\n".join(feedback) if feedback else "✅ Great job! Your grammar looks solid."

# =======================
# 6. Main App Logic
# =======================

# File uploader widget
uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3, or FLAC)", type=["wav", "mp3", "flac"])

# Process button
if uploaded_file is not None:
    st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
    
    if st.button("Process Audio"):
        with st.spinner("Processing audio..."):
            # Process the audio
            audio_data, sample_rate, error = process_audio(uploaded_file)
            
            if error:
                st.error(f"Error processing audio: {error}")
            else:
                st.success("Audio processed successfully!")
                
                # Create tabs for results
                tab1, tab2 = st.tabs(["Results", "Details"])
                
                with st.spinner("Transcribing..."):
                    # Transcribe audio
                    whisper_result = transcribe_audio(audio_data, sample_rate)
                    transcription = whisper_result["text"]
                
                with st.spinner("Analyzing grammar..."):
                    # Analyze grammar
                    language_tool_errors = analyze_with_languagetool(transcription)
                    spacy_analysis = analyze_with_spacy(whisper_result)
                    
                    # Calculate score and generate feedback
                    score = calculate_score(transcription, language_tool_errors, spacy_analysis)
                    feedback = generate_feedback(transcription, language_tool_errors, spacy_analysis)
                
                # Display results in the tabs
                with tab1:
                    # Score display with color based on score
                    score_color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                    st.markdown(f"## 📊 Grammar Score: <span style='color:{score_color}'>{score} / 100</span>", unsafe_allow_html=True)
                    
                    # Transcription
                    st.markdown("### 📝 Transcription:")
                    st.write(transcription)
                    
                    # Feedback
                    st.markdown("### 💡 Feedback:")
                    st.markdown(feedback)
                
                with tab2:
                    # More detailed analysis
                    st.markdown("### 🔍 Detailed Analysis")
                    
                    # Grammar errors
                    st.markdown(f"**Grammar Issues:** {len(language_tool_errors)}")
                    
                    # Sentence structure
                    st.markdown(f"**Average Words per Sentence:** {spacy_analysis['avg_words_per_sentence']:.2f}")
                    
                    # Vocabulary complexity
                    st.markdown(f"**Vocabulary Complexity:** {spacy_analysis['complexity_ratio']:.2f}")
                    
                    # Display sample visualization
                    if len(language_tool_errors) > 0:
                        st.markdown("### Error Distribution")
                        error_types = {}
                        for error in language_tool_errors:
                            error_type = error["error"]
                            if error_type in error_types:
                                error_types[error_type] += 1
                            else:
                                error_types[error_type] = 1
                        
                        # Simple bar chart of error types
                        st.bar_chart(error_types)

# Add information about the app
with st.expander("About this app"):
    st.markdown("""
    ### Grammar Scoring Engine
    
    This application analyzes spoken language by:
    1. Processing uploaded audio files
    2. Transcribing speech to text
    3. Analyzing grammar and structure
    4. Providing a score and actionable feedback
    
    **Note:** This is a demo application. In a production environment, it would use:
    - Real speech-to-text models like OpenAI's Whisper
    - Grammar checking APIs like LanguageTool
    - NLP libraries like spaCy for deeper language analysis
    
    The current demo simulates these components for demonstration purposes.
    """)

# Add sidebar with additional options
with st.sidebar:
    st.header("Settings")
    
    st.subheader("Analysis Options")
    check_grammar = st.checkbox("Check Grammar", value=True)
    check_vocabulary = st.checkbox("Analyze Vocabulary", value=True)
    check_structure = st.checkbox("Evaluate Sentence Structure", value=True)
    
    st.subheader("Display Options")
    show_transcription = st.checkbox("Show Transcription", value=True)
    show_details = st.checkbox("Show Detailed Analysis", value=True)
    
    st.markdown("---")
    st.markdown("Made with ❤️ by Grammar Scoring Engine")

