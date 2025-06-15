import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import json
import random
import os

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("model", local_files_only=True)
tokenizer = BertTokenizer.from_pretrained("model", local_files_only=True)
model.eval()

# Load intent mapping
label2intent = {
    0: "tips", 1: "oops", 2: "dbms", 3: "cn", 4: "os", 5: "quiz", 6: "explain", 7: "track"
}

# Load data files
with open("quiz_data.json", "r", encoding="utf-8") as f:
    quiz_data = json.load(f)

with open("topic_explanations.json", "r", encoding="utf-8") as f:
    topic_explanations = json.load(f)

with open("hr_data.json", "r", encoding="utf-8") as f:
    hr_questions = json.load(f)

user_data_path = "user_data.json"
if os.path.exists(user_data_path):
    with open(user_data_path, "r", encoding="utf-8") as f:
        user_data = json.load(f)
else:
    user_data = {}

def get_notes(topic):
    path = f"{topic.lower()}_notes.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "Notes not found."

def explain_topic(term):
    for topic, content in topic_explanations.items():
        if term.lower() in content:
            return content[term.lower()]
    return "Explanation not found."

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)
    intent_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][intent_id].item()
    return label2intent[intent_id], confidence

def ask_quiz(topic, username, count=3):
    if topic not in quiz_data:
        return "Invalid topic."
    questions = random.sample(quiz_data[topic], min(count, len(quiz_data[topic])))
    score = 0
    for q in questions:
        st.write(f"**Q:** {q['question']}")
        choice = st.radio("Choose your answer:", q['options'], key=q['question'])
        if st.button("Submit", key=f"submit_{q['question']}"):
            if choice == q['answer']:
                st.success("✅ Correct!")
                score += 1
            else:
                st.error(f"❌ Wrong. Correct: {q['answer']}")

    # Store user score
    if username not in user_data:
        user_data[username] = {}
    if topic not in user_data[username]:
        user_data[username][topic] = {"score": 0, "attempts": 0}
    user_data[username][topic]["score"] += score
    user_data[username][topic]["attempts"] += len(questions)

    with open(user_data_path, "w", encoding="utf-8") as f:
        json.dump(user_data, f, indent=2)

    st.write(f"📊 Your score: {score}/{len(questions)}")

# Streamlit interface
st.set_page_config(page_title="Interview Help Chatbot", layout="centered")
st.title("🤖 Interview Help Chatbot")

username = st.text_input("Enter your name to begin:", key="user")

if username:
    query = st.text_input("You:", key="input")
    if query:
        intent, conf = predict_intent(query)

        if intent == "quiz":
            topic = st.selectbox("Select topic for quiz:", list(quiz_data.keys()))
            num_q = st.slider("How many questions?", 1, 5, 3)
            if st.button("Start Quiz"):
                ask_quiz(topic, username, num_q)

        elif intent == "explain":
            topic = st.text_input("What topic should I explain?")
            if topic:
                st.info(explain_topic(topic))

        elif intent == "tips":
            st.info("💡 Be confident, revise your resume, and practice core CS + DSA.")

        elif intent == "track":
            if username in user_data:
                for t, stats in user_data[username].items():
                    st.write(f"**{t.upper()}**: {stats['score']} / {stats['attempts']} correct")
            else:
                st.write("No progress yet recorded.")

        elif intent == "oops":
            st.write("OOPs includes classes, inheritance, polymorphism, encapsulation.")
        elif intent == "os":
            st.write("OS topics include scheduling, deadlocks, threads, semaphores.")
        elif intent == "dbms":
            st.write("DBMS includes normalization, transactions, indexing, SQL.")
        elif intent == "cn":
            st.write("CN covers OSI model, IP addressing, routing, protocols.")
        else:
            st.write(f"(Intent: {intent}, Confidence: {conf:.2f}) Sorry, I didn't get that.")
