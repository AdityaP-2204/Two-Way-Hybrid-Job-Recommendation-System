# Two-Way-Hybrid-Job-Recommendation-System

A **high-performance, two-way recommendation system** built for a college course project.  
It uses a **hybrid neural network** to match **job candidates to job postings** — and, in reverse, to recommend **relevant jobs to candidates**.

The model is trained on a synthetic dataset of users, jobs, and their interactions, achieving an impressive:

- **AUC:** `0.992`
- **NDCG@10:** `0.997`

---

## 🚀 Core Features

### 🔹 Hybrid Model (NeuMF-style)
Fuses two powerful techniques into one neural architecture:

- **Collaborative Filtering (GMF):**  
  Learns hidden behavioral patterns from user–job interaction history  
  (e.g., *who got shortlisted for what*).

- **Content-Based Filtering (MLP):**  
  Learns logical matching rules from explicit features  
  (e.g., *user_experience > job_experience*, *skill matching*, *salary alignment*).

---

### 🔹 2-Way Recommendations
The same trained model can be used for both directions:

- **For Job Seekers:** Recommend the best jobs for a given user.  
- **For Companies:** Recommend the top candidates for a given job.

---

### 🔹 Cold-Start Problem Solved
Even with **no prior interaction data**, the MLP (content-based) path can generate **high-quality recommendations** using only profile and job features.

---

## 🧠 Model Architecture

A **two-tower neural network**, where each tower learns from a different type of signal.

### 🏗️ Path A: GMF (Collaborative)
- **Input:** `user_id`, `job_id`  
- **Process:**  
  Each ID is embedded to form latent “profile” vectors, then **multiplied** to generate a `collaborative_signal`.  
- **Learns:** Behavioral and latent user-job similarity patterns.

### 🧩 Path B: MLP (Content)
- **Input:** All explicit features (skills, experience, salary, etc.)  
- **Process:**  
  The combined feature vector passes through stacked `Dense` layers to produce a `content_signal`.  
- **Learns:** Logical matching rules and semantic relationships.

### 🔗 Fusion Layer
- Concatenates both signals: `[collaborative_signal, content_signal]`
- Passes through a final `Dense` prediction layer → outputs the **shortlist probability** (0.0 to 1.0)

---

## 💾 Dataset

The system is trained on **three synthetic CSV files**:

| File | Description |
|------|--------------|
| **`users.csv`** | Contains `user_id` and features like experience, salary expectation, and skill proficiencies (`Python_proficiency`, `SQL_proficiency`, etc.) |
| **`jobs.csv`** | Contains `job_id` and features like experience required, salary range, and expected skill levels |
| **`interactions.csv`** | Contains `(user_id, job_id, shortlisted)` triplets — the ground truth for supervised training, synthetically balanced (~40% positive, 60% negative) |

---

## ⚙️ Usage

The main script **`model.py`** handles **all preprocessing, training, and evaluation**.

### 🧩 1. Training the Model

```bash
python model.py
