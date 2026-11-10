# ⚡ SMARA - Smart Resume Assistant 📄🤖

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B.svg)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/AI-Gemini_3.5_Flash_Lite-8E75B2.svg)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SMARA (Smart Resume Assistant)** is a high-performance, domain-agnostic Applicant Tracking System (ATS) optimizer. It evaluates candidate resumes against target job descriptions using a **Hybrid Scoring Engine** (TF-IDF Contextual Similarity + LLM Domain Competency Extraction) and provides actionable, non-hallucinated AI coaching to bridge skill gaps.

---

## ✨ Key Features

*   **🧠 Hybrid ATS Scoring System**: Moves beyond brittle keyword matching. Calculates a weighted score using 40% TF-IDF contextual similarity and 60% LLM-extracted hard-skill overlap.
*   **🎯 Domain-Agnostic Extraction**: Works across Software, Finance, Marketing, and HR. Automatically filters out corporate noise (e.g., "Barclays", "Bengaluru", "salary") to focus strictly on measurable competencies.
*   **🤖 SMARA AI Coach**: Powered by **Gemini 3.5 Flash Lite**, providing 3 highly targeted, strategic bullet-point rewrite suggestions to naturally incorporate missing skills without fabricating experience.
*   **🔐 Bring Your Own Key (BYOK) Architecture**: Enterprise-ready security. Automatically loads from `.env` or Streamlit Secrets, with a seamless manual fallback UI for users.
*   **💾 Stateful Multi-Download Engine**: Built with Streamlit `st.session_state` to allow users to download multiple text reports and visual PNG charts without triggering app reruns or data loss.
*   **🛡️ Advanced String Normalization**: Custom Regex pipelines ensure that variations like `React.js` vs `react` or `CI/CD` vs `ci cd` are perfectly matched.

---

## 🏗️ Architecture & Tech Stack

*   **Frontend & UI**: [Streamlit](https://streamlit.io/) (with custom CSS for Hero Cards and Skill Badges)
*   **Text Processing & NLP**: `scikit-learn` (TF-IDF Vectorization, Cosine Similarity), `pdfplumber` (PDF Text Extraction), `re` (Regex Normalization).
*   **Generative AI**: `google-genai` SDK (Gemini 3.5 Flash Lite for high RPM, low-latency extraction and coaching).
*   **Data Visualization**: `seaborn`, `matplotlib`, `pandas`.

---

## 🚀 Installation & Setup

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mdshibanalam/smart-resume-assistant.git
   cd smart-resume-assistant
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure API Keys:**
   Create a `.streamlit/secrets.toml` file or a `.env` file in the root directory:
   ```toml
   # .streamlit/secrets.toml
   GEMINI_API_KEY = "your_google_gemini_api_key_here"
   ```

4. **Run SMARA:**
   ```bash
   streamlit run app.py
   ```

### Cloud Deployment (Streamlit Cloud)
1. Push this repository to GitHub.
2. Connect the repo to [Streamlit Community Cloud](https://share.streamlit.io/).
3. In the deployment settings, select **Python 3.11**.
4. Navigate to **Advanced Settings -> Secrets** and paste your API key:
   ```toml
   GEMINI_API_KEY = "your_google_gemini_api_key_here"
   ```

---

## 💡 How to Use

1. **Upload** your current Resume in PDF format.
2. **Paste** the target Job Description into the text area.
3. **Analyze**: SMARA will automatically extract the core skills, normalize the strings, and calculate your exact overlap.
4. **Export**: Download your detailed Skill Gap Audit (`.txt`), Coverage Chart (`.png`), and AI Strategy Report (`.txt`) directly from the dashboard.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/mdshibanalam/smart-resume-assistant/issues).

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.