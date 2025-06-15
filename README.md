# 🤖 Interview Help Chatbot

A conversational AI-powered command-line chatbot designed to help students and job seekers prepare for technical and HR interviews. Built with a fine-tuned **BERT model** and supports a complete interview prep suite including quizzes, explanations, notes, and HR guidance.

---

## 🌟 Features

### 🔍 Smart Intent Detection
- Fine-tuned BERT model to classify intents like `quiz`, `notes`, `explain`, `tips`, `track`, etc.
- Understands natural queries like:
  - `quiz dbms 3`
  - `explain polymorphism`
  - `notes os`
  - `give me HR questions`

### 🧠 Topic-wise Interview Prep
- Supports **Operating Systems, DBMS, Computer Networks, OOPs**
- Choose to:
  - Take quizzes (MCQs)
  - View quick notes
  - Get explanations of specific topics

### 📘 Technical Notes
- Clean summaries of concepts from each subject
- Stored in `os_notes.txt`, `dbms_notes.txt`, etc.
- Easily extendable for new topics

### ❓ Randomized Quiz Generator
- Quiz data stored in `quiz_data.json`
- Select topic and number of questions
- Tracks score + attempts per user in `user_data.json`
- Ensures no repetition within a session

### 🧾 Topic Explanations
- Stored in `topic_explanations.json`
- User can ask `explain deadlock` or `explain inheritance`
- Can be expanded to support more concepts

### 💬 HR Interview Mode
- Behavioral questions pulled from `hr_data.json`
- Shows 10 initial questions → ask for more
- Questions only (no metadata clutter)

### 💾 Session Memory
- Tracks user quiz progress, score, and status in `user_data.json`
- Resumes seamlessly across sessions

---

## 📂 Folder Structure
```
interview_help_chatbot/
├── chatbot.py                    # Main terminal chatbot
├── model/                        # Fine-tuned BERT model + config
├── quiz_data.json                # MCQs by topic
├── intents_data.csv              # Mapped intents with their labels
├── topic_explanations.json       # Concept explanations
├── user_data.json                # Persistent user progress
├── hr_data.json                  # HR questions
├── os_notes.txt / dbms_notes.txt / ...
├── requirements.txt              # Python dependencies
└── README.md
```

## 🛠️ Tech Stack
- Python 3.10+
- Hugging Face Transformers
- BERT (bert-base-uncased)
- PyTorch
- JSON storage (no DBs)

--- 

## 🧾 Here is a demo walkthrough of the output:

🤖 Interview Help Chatbot
==============================
Enter your name: deepti
Welcome deepti! 🎉
Ready to ace your interviews? Let's get started!

💬 What would you like to do?
1. Topic-wise Preparation
2. Explain a Topic
3. HR Interview Preparation
4. Interview Tips
5. View Progress
6. Update Track
7. Exit
Enter choice (1-7): 1

🧠 Choose a subject: os / dbms / cn / oops
Topic: dbms
What would you like to do?
a. Quiz
b. Explain Topic
c. Quick Notes
Enter choice (a/b/c): a
How many questions do you want?: 2

❓ What is normalization in DBMS?
1. Converting programming logic into database schema
2. Removing redundancy and improving data integrity
3. Creating a schema using SQL
4. Backing up databases
Your answer (enter option number): 2
✅ Correct!

❓ What is the purpose of a primary key?
1. Allow null values
2. Uniquely identify a row
3. Create relationships between tables
4. Backup a table
Your answer (enter option number): 2
✅ Correct!

📊 Final Score for dbms: 2 / 2

💬 What would you like to do?
1. Topic-wise Preparation
2. Explain a Topic
3. HR Interview Preparation
4. Interview Tips
5. View Progress
6. Update Track
7. Exit
Enter choice (1-7): 2

📘 Topic Explanation
💡 Tip: You can ask about topics like 'TCP vs UDP', 'normalization', 'deadlock', etc.

What would you like to do?
1. Explain a specific topic
2. Show all available topics
3. Back to main menu
Enter choice (1-3): 1
Which topic do you want explained? (or 'back' to return): deadlock

📘 Explanation of DEADLOCK:
A deadlock is a situation where a set of processes are blocked because each process is holding a resource and waiting for another.

💬 What would you like to do?
1. Topic-wise Preparation
2. Explain a Topic
3. HR Interview Preparation
4. Interview Tips
5. View Progress
6. Update Track
7. Exit
Enter choice (1-7): 3

🧑‍💼 Here are some important HR questions to prepare:
- Why should we hire you?
- What are your strengths and weaknesses?
- Describe a time you worked under pressure.
- How do you handle feedback?
- What motivates you?
- What are your goals?
- Why do you want to join our company?
- Describe a challenging project you worked on.
- Are you a team player?
- Tell me about yourself.

Do you want more HR questions? (yes/no): no

💬 What would you like to do?
1. Topic-wise Preparation
2. Explain a Topic
3. HR Interview Preparation
4. Interview Tips
5. View Progress
6. Update Track
7. Exit
Enter choice (1-7): 5

📊 Progress Report for deepti:
========================================
📚 DBMS: 2/2 (100.0%)
📈 Topics covered: 1/4 subjects
🎯 Overall: 2/2 (100.0%)

💡 Recommendations:
🌟 Excellent! You're doing great. Keep practicing!

💬 What would you like to do?
1. Topic-wise Preparation
...
