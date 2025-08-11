# app.py
import streamlit as st
import pandas as pd
import os
from utils.extract_foods_nlp import extract_food_items
from utils.transcribe_audio import transcribe_audio_input
# Removed TTS playback per user request

# Optional mic recorder component
MIC_AVAILABLE = False
try:
    from streamlit_mic_recorder import mic_recorder  # type: ignore
    MIC_AVAILABLE = True
except Exception:
    MIC_AVAILABLE = False
 

# Load dataset (robust path + error handling)
DATASET_PATH = os.path.join(os.path.dirname(__file__), "Diseases_Foods_Dataset_Real.csv")
df = None
_dataset_load_error = None
try:
    df = pd.read_csv(DATASET_PATH)
except Exception as e:
    _dataset_load_error = str(e)

def get_recommendations(disease, food_list):
    results = []
    disease_norm = disease.lower().strip()

    # Find row for disease
    disease_rows = df[df['Disease'].str.lower().str.strip() == disease_norm]
    if disease_rows.empty:
        for food in food_list:
            results.append({
                'Food': food,
                'Recommendation': 'Unknown',
                'Reason': 'Disease not found in dataset',
                'Calories': '-',
                'Alternative': '-'
            })
        return results

    row = disease_rows.iloc[0]
    not_rec_str = str(row.get('Foods/Ingredients Not Recommended', '') or '')
    rec_str = str(row.get('Recommended Foods', '') or '')
    total_calories_not_rec = row.get('Total Calories of Not Recommended Foods', '-')

    def normalize_list(s):
        return [item.strip().lower() for item in s.split(',') if item.strip()]

    not_rec_items = set(normalize_list(not_rec_str))
    rec_items = set(normalize_list(rec_str))

    def matches(food_name, items_set):
        f = food_name.lower().strip()
        if f in items_set:
            return True
        # partial contains match
        return any((f in item) or (item in f) for item in items_set)

    for food in food_list:
        if matches(food, not_rec_items):
            status = 'Not Recommended'
            reason = 'Found in Not Recommended list for this disease'
            cal = total_calories_not_rec if pd.notna(total_calories_not_rec) else '-'
            alternative = ', '.join(sorted(rec_items)) if rec_items else '-'
        elif matches(food, rec_items):
            status = 'Recommended'
            reason = 'Found in Recommended list for this disease'
            cal = '-'
            alternative = '-'
        else:
            status = 'Unknown'
            reason = 'Food not found in dataset lists for this disease'
            cal = '-'
            alternative = ', '.join(sorted(rec_items)) if rec_items else '-'

        results.append({
            'Food': food,
            'Recommendation': status,
            'Reason': reason,
            'Calories': cal,
            'Alternative': alternative
        })
    return results

st.set_page_config(page_title="Healthy Food Monitor", layout="centered")
st.title("🥗 Healthy Food Intake Monitor")
st.markdown("Enter or speak your disease and foods to get a personalized recommendation.")

if _dataset_load_error:
    st.error(f"Failed to load dataset from {DATASET_PATH}: {_dataset_load_error}")

# Disease input section
st.subheader("1. Enter Your Disease")
if 'disease_text' not in st.session_state:
    st.session_state['disease_text'] = ''
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.session_state['disease_text'] = st.text_input("Type your disease (e.g., Diabetes, Hypertension)", value=st.session_state['disease_text'])
with col2:
    audio_disease = st.file_uploader("Upload voice (Disease)", type=["wav", "mp3"], key="disease_voice")
    if audio_disease:
        text = transcribe_audio_input(audio_disease)
        if text:
            st.session_state['disease_text'] = text
            st.success(f"Recognized Disease: {text}")
with col3:
    if MIC_AVAILABLE:
        st.caption("Or record via mic")
        disease_mic = mic_recorder(key="disease_mic", start_prompt="Record", stop_prompt="Stop", just_once=True)
        if disease_mic and disease_mic.get("bytes"):
            text = transcribe_audio_input(disease_mic["bytes"])  # type: ignore[index]
            if text:
                st.session_state['disease_text'] = text
                st.success(f"Recognized Disease (Mic): {text}")
    else:
        st.caption("Install streamlit-mic-recorder for mic recording")

# Food input section
st.subheader("2. Provide Foods (Text / Voice)")
if 'food_text' not in st.session_state:
    st.session_state['food_text'] = ''
food_text_input = st.text_input("Type food items (comma separated)", value=st.session_state['food_text'])
st.session_state['food_text'] = food_text_input
audio_food = st.file_uploader("Upload voice (Food)", type=["wav", "mp3"], key="food_voice")

# Optional: record voice live
st.caption("Prefer voice? Record directly from your mic:")
mic_record = None
if MIC_AVAILABLE:
    mic_record = mic_recorder(key="food_mic", start_prompt="Record", stop_prompt="Stop", just_once=True)
elif hasattr(st, 'experimental_audio_input'):
    mic_record = st.experimental_audio_input("Hold to record or click to toggle", help="Requires browser microphone permission")

food_items = []
if st.session_state.get('food_text'):
    food_items = [f.strip() for f in extract_food_items(st.session_state['food_text'])]
elif audio_food:
    transcript = transcribe_audio_input(audio_food)
    st.success(f"Recognized Foods: {transcript}")
    food_items = [f.strip() for f in extract_food_items(transcript)]
elif mic_record is not None:
    mic_bytes = None
    # streamlit-mic-recorder returns a dict with 'bytes'
    if isinstance(mic_record, dict) and mic_record.get('bytes'):
        mic_bytes = mic_record['bytes']  # type: ignore[index]
    elif hasattr(mic_record, 'getvalue'):
        # experimental_audio_input returns a BytesIO-like object
        mic_bytes = mic_record.getvalue()
    if mic_bytes:
        transcript = transcribe_audio_input(mic_bytes)
        if transcript:
            st.success(f"Recognized Foods (Mic): {transcript}")
            food_items = [f.strip() for f in extract_food_items(transcript)]
 
 

# Show recommendations
if _dataset_load_error:
    st.stop()
elif st.session_state.get('disease_text') and food_items:
    st.subheader("3. Food Recommendations")
    recs = get_recommendations(st.session_state['disease_text'], food_items)
    for rec in recs:
        st.markdown(f"### 🍽️ {rec['Food']}")
        st.write(f"**Recommendation:** {rec['Recommendation']}")
        st.write(f"**Reason:** {rec['Reason']}")
        st.write(f"**Calories:** {rec['Calories']}")
        st.write(f"**Alternative:** {rec['Alternative']}")
        st.markdown("---")
else:
    st.info("Please provide both disease and food inputs.")