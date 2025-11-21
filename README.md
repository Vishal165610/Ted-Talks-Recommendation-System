🎥 TED Talks Recommendation System
📌 Project Overview

This is a simple AI/ML project that recommends TED Talks based on what the user types. It uses modern Natural Language Processing so the system understands the meaning of the text, not just keywords.

Perfect for Semester 1 internal assessment and easy to explain in viva.

🎯 Objective

To recommend relevant TED Talks using:
AI-based text understanding
Sentence embeddings
Similarity matching

🧠 How It Works (Very Simple)

TED Talk data is loaded from a CSV file.
Text (title + speaker + description + tags) is combined.
This text is converted into numerical vectors using SentenceTransformer (SBERT).
When a user enters a query, it is also converted into a vector.
Cosine Similarity compares both vectors.
The system shows the top 5 most similar TED Talks.

Example: User types: "motivation and success" System shows: TED Talks related to confidence, mindset, and growth.

🗂 Project Structure
<img width="353" height="366" alt="image" src="https://github.com/user-attachments/assets/465b308b-b941-424d-8351-10d84aef8032" />




⚙ Technologies Used
Python
Pandas
Streamlit
SentenceTransformers
Scikit-learn

▶ How To Run The Project
Install Python
Open terminal in project folder

Install dependencies:
pip install -r requirements.txt

Run the app:

python -m streamlit run app.py
Open the browser link shown

🧪 Behind The Scenes (Simple Explanation)

SentenceTransformer converts text into smart vectors
These vectors store the meaning of sentences
Cosine Similarity checks which talks are most close in meaning
The system ranks and displays the best matches
This approach is more advanced and more accurate than TF-IDF.

✅ Features

Chat-based recommendation
Meaning-based search
Fast results
Clean UI
Beginner-friendly AI logic

🎓 How To Explain In Viva
"This project recommends TED Talks using AI-based text embeddings. It uses SentenceTransformer to convert text into numerical vectors and Cosine Similarity to find the most similar talks. This method understands the meaning of the input, making recommendations more accurate than traditional TF-IDF."


This project demonstrates a modern recommendation system using semantic understanding. It is simple, effective, and perfect for first-year engineering students to understand how AI can recommend content based on meaning.

