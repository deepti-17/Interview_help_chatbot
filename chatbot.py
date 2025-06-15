import torch
import json
import random
import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
from fuzzywuzzy import fuzz

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("model", local_files_only=True, ignore_mismatched_sizes=True)
tokenizer = BertTokenizer.from_pretrained("model", local_files_only=True)
model.eval()

# Load data
with open("quiz_data.json", "r") as f:
    quiz_data = json.load(f)

with open("hr_data.json", "r") as f:
    hr_questions = json.load(f)

with open("topic_explanations.json", "r") as f:
    topic_explanations = json.load(f)

user_data_path = "user_data.json"
if os.path.exists(user_data_path):
    with open(user_data_path, "r") as f:
        user_data = json.load(f)
else:
    user_data = {}

label2intent = {
    0: "tips",
    1: "oops",
    2: "dbms",
    3: "cn",
    4: "os",
    5: "quiz",
    6: "explain",
    7: "track",
}

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    intent_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][intent_id].item()
    return label2intent[intent_id], confidence

def read_notes(topic):
    file_path = f"{topic}_notes.txt"
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            return f"⚠️ Error reading notes: {e}"
    return "Notes not available."

def find_best_topic_match(user_input, topic_explanations):
    """
    Find the best matching topic in nested structure using multiple strategies:
    1. Exact match (case-insensitive) 
    2. Partial matching
    3. Fuzzy string matching
    4. Category matching
    """
    user_input = user_input.lower().strip()
    best_match = None
    best_score = 0
    best_category = None
    
    # Search through all categories and their topics
    for category, topics in topic_explanations.items():
        # First check if user is asking for entire category
        if user_input == category.lower():
            return category, topics, 100
            
        # Then search within each topic in the category
        for topic_key, topic_desc in topics.items():
            topic_lower = topic_key.lower()
            
            # Strategy 1: Exact match
            if user_input == topic_lower:
                return topic_key, {topic_key: topic_desc}, 100
            
            # Strategy 2: Partial match
            if user_input in topic_lower or topic_lower in user_input:
                if 90 > best_score:
                    best_match = topic_key
                    best_score = 90
                    best_category = {topic_key: topic_desc}
            
            # Strategy 3: Fuzzy matching
            scores = [
                fuzz.ratio(user_input, topic_lower),
                fuzz.partial_ratio(user_input, topic_lower), 
                fuzz.token_sort_ratio(user_input, topic_lower),
                fuzz.token_set_ratio(user_input, topic_lower)
            ]
            max_score = max(scores)
            
            if max_score > best_score and max_score > 60:
                best_score = max_score
                best_match = topic_key
                best_category = {topic_key: topic_desc}
    
    # Strategy 4: Smart keyword matching for common variations
    keyword_mappings = {
        'tcp': ['tcp', 'transmission control protocol'],
        'udp': ['udp', 'user datagram protocol'], 
        'tcp vs udp': ['tcp vs udp', 'udp vs tcp', 'tcp udp', 'udp tcp'],
        'polymorphism': ['polymorphism', 'method overloading overriding'],
        'sql': ['sql', 'structured query language'],
        'deadlock': ['deadlock', 'dead lock'],
        'process': ['process vs thread', 'process thread'],
        'normalization': ['normalization', 'normal forms', '1nf 2nf 3nf']
    }
    
    for target, variations in keyword_mappings.items():
        if any(var in user_input for var in variations):
            # Search for the target in all topics
            for category, topics in topic_explanations.items():
                for topic_key, topic_desc in topics.items():
                    if target in topic_key.lower() or any(var in topic_key.lower() for var in variations):
                        return topic_key, {topic_key: topic_desc}, 85
    
    return best_match, best_category, best_score

def explain_topic(topic):
    """
    Enhanced topic explanation with fuzzy matching for nested structure
    """
    if not topic.strip():
        print("⚠️ Please enter a topic to explain.")
        return
    
    # Try to find best match
    best_match, explanation_data, confidence = find_best_topic_match(topic, topic_explanations)
    
    if confidence >= 70 and explanation_data:  # Good match found
        print(f"\n📘 Explanation of {best_match.upper()}:")
        
        # Handle different explanation formats
        if isinstance(explanation_data, dict):
            if len(explanation_data) == 1:  # Single topic
                for subtopic, desc in explanation_data.items():
                    print(f"{desc}")
            else:  # Multiple topics (category)
                for subtopic, desc in explanation_data.items():
                    print(f"\n🔹 {subtopic.title()}: {desc}")
        elif isinstance(explanation_data, str):
            print(explanation_data)
        else:
            print(explanation_data)
            
        if confidence < 100:
            print(f"\n💡 (Showing results for '{best_match}' - closest match to '{topic}')")
    
    else:  # No good match found
        print(f"❌ Explanation for '{topic}' not available.")
        
        # Suggest similar topics from all categories
        suggestions = []
        for category, topics in topic_explanations.items():
            for topic_key in topics.keys():
                score = fuzz.partial_ratio(topic.lower(), topic_key.lower())
                if score > 40:  # Reasonable similarity
                    suggestions.append((topic_key, score, category))
        
        if suggestions:
            suggestions.sort(key=lambda x: x[1], reverse=True)
            print("\n💡 Did you mean one of these topics?")
            for suggestion, _, cat in suggestions[:5]:  # Show top 5 suggestions
                print(f"   • {suggestion} ({cat.upper()})")
        else:
            print("\n📚 Try asking about topics like:")
            print("   • udp vs tcp (networking)")
            print("   • polymorphism (OOP)")  
            print("   • normalization (DBMS)")
            print("   • deadlock (OS)")
            print("\nOr type 'show topics' to see all available topics.")

def show_all_topics():
    """
    Display all available topics organized by category 
    """
    print("\n📚 All Available Topics:")
    
    category_icons = {
        'oops': '🔷 Object-Oriented Programming',
        'dbms': '🗄️ Database Management Systems', 
        'os': '💻 Operating Systems',
        'cn': '🌐 Computer Networks'
    }
    
    for category, topics in topic_explanations.items():
        icon_name = category_icons.get(category, f"📘 {category.upper()}")
        print(f"\n{icon_name}:")
        
        # Sort topics alphabetically and display
        for topic in sorted(topics.keys()):
            print(f"   • {topic}")
        
        print(f"   ({len(topics)} topics available)")

def handle_explain_choice():
    """
    Enhanced handler for topic explanation with better UX
    """
    print("\n📘 Topic Explanation")
    print("💡 Tip: You can ask about topics like 'TCP vs UDP', 'normalization', 'deadlock', etc.")
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Explain a specific topic")
        print("2. Show all available topics")
        print("3. Back to main menu")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            topic = input("Which topic do you want explained? (or 'back' to return): ").strip()
            if topic.lower() == 'back':
                break
            explain_topic(topic)
            
        elif choice == "2":
            show_all_topics()
            
        elif choice == "3":
            break
            
        else:
            print("⚠️ Invalid choice. Please enter 1, 2, or 3.")

def ask_quiz(username, topic, num_questions):
    questions = random.sample(quiz_data[topic], min(num_questions, len(quiz_data[topic])))
    correct_count = 0

    for q in questions:
        print(f"\n❓ {q['question']}")
        for idx, opt in enumerate(q["options"], 1):
            print(f"{idx}. {opt}")
        ans = input("Your answer (enter option number): ").strip()

        if ans.isdigit():
            selected_idx = int(ans) - 1
            if 0 <= selected_idx < len(q["options"]):
                selected = q["options"][selected_idx].strip().lower()
                correct = q["answer"].strip().lower()
                is_correct = selected == correct

                if username not in user_data:
                    user_data[username] = {}
                if topic not in user_data[username] or not isinstance(user_data[username][topic], dict):
                    user_data[username][topic] = {"score": 0, "attempts": 0}
                else:
                    user_data[username][topic].setdefault("score", 0)
                    user_data[username][topic].setdefault("attempts", 0)


                user_data[username][topic]["attempts"] += 1

                if is_correct:
                    user_data[username][topic]["score"] += 1
                    correct_count += 1
                    print("✅ Correct!")
                else:
                    print(f"❌ Incorrect. Correct answer: {q['answer']}")
            else:
                print("⚠️ Invalid option number.")
        else:
            print("⚠️ Invalid input.")

    with open(user_data_path, "w") as f:
        json.dump(user_data, f, indent=2)

    print(f"\n📊 Final Score for {topic}: {correct_count} / {len(questions)}\n")


def handle_hr_questions():
    print("\n🧑‍💼 Here are some important HR questions to prepare:")
    asked = random.sample(hr_questions, min(10, len(hr_questions)))
    for q in asked:
        print(f"- {q['question']}")
    while True:
        more = input("\nDo you want more HR questions? (yes/no): ").strip().lower()
        if more == "yes":
            more_qs = random.sample(hr_questions, min(5, len(hr_questions)))
            for q in more_qs:
                print(f"- {q['question']}")
        else:
            break

def show_user_progress(username):
    """
    Display user's learning progress and recommendations
    """
    if username not in user_data or not user_data[username]:
        print(f"\n📊 No progress data found for {username}")
        print("💡 Start taking quizzes to track your progress!")
        return
    
    print(f"\n📊 Progress Report for {username}:")
    print("=" * 40)
    
    total_topics = 0
    total_score = 0
    total_attempts = 0
    
    for topic, data in user_data[username].items():
        if isinstance(data, dict) and "score" in data:
            score = data.get("score", 0)
            attempts = data.get("attempts", 0)
            if attempts > 0:
                accuracy = (score / attempts) * 100
                print(f"📚 {topic.upper()}: {score}/{attempts} ({accuracy:.1f}%)")
                total_topics += 1
                total_score += score
                total_attempts += attempts
        else:
            print(f"✅ {topic.upper()}: Completed")
    
    if total_attempts > 0:
        overall_accuracy = (total_score / total_attempts) * 100
        print("=" * 40)
        print(f"🎯 Overall: {total_score}/{total_attempts} ({overall_accuracy:.1f}%)")
        
        # Provide recommendations
        print("\n💡 Recommendations:")
        if overall_accuracy >= 80:
            print("🌟 Excellent! You're doing great. Keep practicing!")
        elif overall_accuracy >= 60:
            print("👍 Good progress! Focus on weaker topics.")
        else:
            print("📖 Keep studying! Practice more on challenging topics.")
    
    print(f"\n📈 Topics covered: {total_topics}/4 subjects")

def main_menu(username):
    while True:
        print("\n💬 What would you like to do?")
        print("1. Topic-wise Preparation")
        print("2. Explain a Topic")
        print("3. HR Interview Preparation")
        print("4. Interview Tips")
        print("5. View Progress")
        print("6. Update Track")
        print("7. Exit")
        choice = input("Enter choice (1-7): ")

        if choice == "1":
            print("\n🧠 Choose a subject: os / dbms / cn / oops")
            topic = input("Topic: ").lower()
            if topic not in ["os", "dbms", "cn", "oops"]:
                print("Invalid topic.")
                continue
            print("What would you like to do?")
            print("a. Quiz\nb. Explain Topic\nc. Quick Notes")
            sub_choice = input("Enter choice (a/b/c): ").lower()

            if sub_choice == "a":
                n_input = input("How many questions do you want?: ").strip()
                if n_input.isdigit():
                    n = int(n_input)
                    ask_quiz(username, topic, n)
                else:
                    print("⚠️ Please enter a valid number.")

            elif sub_choice == "b":
                explain_topic(topic)
            elif sub_choice == "c":
                notes = read_notes(topic)
                print(f"\n📓 {topic.upper()} Quick Notes:\n{notes}")
            else:
                print("Invalid choice.")

        elif choice == "2":
            handle_explain_choice()

        elif choice == "3":
            handle_hr_questions()

        elif choice == "4":
            print("💡 Interview Tips:")
            print("🎯 Technical Interviews:")
            print("   • Practice coding problems daily")
            print("   • Understand time/space complexity")
            print("   • Know your data structures & algorithms")
            print("   • Practice system design basics")
            print("\n🤝 HR Interviews:")
            print("   • Know your resume inside out")
            print("   • Prepare STAR method examples")
            print("   • Research the company thoroughly")
            print("   • Be confident and authentic")
            print("\n📚 General Tips:")
            print("   • Mock interviews with friends")
            print("   • Prepare questions to ask them")
            print("   • Get good sleep before the interview")

        elif choice == "5":
            show_user_progress(username)

        elif choice == "6":
            subject = input("Which topic did you complete?: ").lower()
            if username not in user_data:
                user_data[username] = {}
            user_data[username][subject] = {"status": "completed"}
            with open(user_data_path, "w") as f:
                json.dump(user_data, f, indent=2)
            print(f"✅ Progress updated for {subject}.")

        elif choice == "7":
            print("👋 Goodbye! Good luck with your interviews.")
            print("🌟 Remember: Practice makes perfect!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    print("🤖 Interview Help Chatbot")
    print("=" * 30)
    username = input("Enter your name: ").strip()
    print(f"Welcome {username}! 🎉")
    print("Ready to ace your interviews? Let's get started!")
    main_menu(username)