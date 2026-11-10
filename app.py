import io
import os
import re
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# Load environment variables from local .env file (if present)
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SMARA - Smart Resume Assistant",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for UI Badges and Hero Cards
st.markdown("""
<style>
    .hero-card {
        background-color: #1E1E2E;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        border: 1px solid #313244;
    }
    .hero-score {
        font-size: 54px;
        font-weight: 800;
        margin: 0;
    }
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        margin-top: 10px;
    }
    .badge-matched {
        background-color: #1E3A2B;
        color: #2ECC71;
        border: 1px solid #2ECC71;
        padding: 6px 12px;
        border-radius: 6px;
        display: inline-block;
        margin: 4px;
        font-size: 13px;
        font-weight: 500;
    }
    .badge-missing {
        background-color: #3A1E1E;
        color: #E74C3C;
        border: 1px solid #E74C3C;
        padding: 6px 12px;
        border-radius: 6px;
        display: inline-block;
        margin: 4px;
        font-size: 13px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# App Branding Header
st.title("⚡ SMARA - Smart Resume Assistant")
st.caption("An Intelligent Applicant Tracking System with An AI-Powered Resume Tailor Assistant - Powered by SMARA AI")

# Sanitized Secret Resolver: Checks .streamlit/secrets.toml -> .env -> system OS env
def get_system_api_key():
    # 1. Check Streamlit secrets (.streamlit/secrets.toml or Streamlit Cloud Secrets UI)
    try:
        if "GEMINI_API_KEY" in st.secrets:
            raw_key = str(st.secrets["GEMINI_API_KEY"]).strip().strip('"').strip("'")
            if raw_key:
                return raw_key
    except Exception:
        pass
    
    # 2. Check environment variables (.env fallback)
    env_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if env_key:
        return env_key
        
    return ""

# Sidebar Configuration & Dynamic BYOK Authentication Fallback
st.sidebar.header("⚙️ AI Configuration")

system_api_key = get_system_api_key()
user_api_key = ""
api_valid = False
gemini_client = None
auth_error_msg = ""

def validate_gemini_key(key):
    if not key:
        return False, None, "No API key provided."
    try:
        client = genai.Client(api_key=key)
        # Lightweight authentication check on app load
        client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents='Ping'
        )
        return True, client, "Authentication Successful"
    except Exception as e:
        return False, None, str(e)

# 1. Attempt verification of system secrets key first
if system_api_key:
    api_valid, gemini_client, auth_error_msg = validate_gemini_key(system_api_key)
    if api_valid:
        st.sidebar.success("SMARA AI Enabled (System Key) ✅")
        user_api_key = system_api_key
    else:
        st.sidebar.warning("⚠️ System Key Auth Failed")
        st.sidebar.caption(f"Reason: {auth_error_msg[:80]}")

# 2. BYOK Fallback: Allow manual key input if system key is missing or invalid
if not api_valid:
    byok_key = st.sidebar.text_input("Gemini API Key (BYOK Fallback)", type="password")
    if byok_key:
        clean_byok = byok_key.strip().strip('"').strip("'")
        api_valid, gemini_client, auth_error_msg = validate_gemini_key(clean_byok)
        if api_valid:
            st.sidebar.success("SMARA AI Enabled (User Key) ✅")
            user_api_key = clean_byok
        else:
            st.sidebar.error("❌ SMARA AI Auth Failed")
            st.sidebar.caption(f"Error: {auth_error_msg[:90]}")

# 3. Help guide displayed in sidebar if authentication fails or key is missing
if not api_valid:
    with st.sidebar.expander("❓ How to get a FREE Gemini API Key", expanded=True):
        st.markdown("""
        1. Go to [Google AI Studio](https://aistudio.google.com/).
        2. Sign in with your Google account.
        3. Click **"Get API key"** and then **"Create API key"**.
        4. Copy your key and paste it above (or save it in `.streamlit/secrets.toml` as `GEMINI_API_KEY`).
        """)

# PDF Extraction Function
def extract_text_from_pdf(pdf_file):
    extracted_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
    return extracted_text.strip()

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\+\#\-\/\s]', ' ', text)
    return " ".join(text.split())

# Normalize skills for uniform string matching
def normalize_skill(skill_str):
    skill_str = skill_str.lower()
    skill_str = re.sub(r'[\.\-\/]', ' ', skill_str)
    return " ".join(skill_str.split())

# Refined LLM Domain Competency Extraction Function
def extract_domain_competencies(jd_text, gemini_client):
    if not gemini_client:
        return []
    
    prompt = f"""
    You are an expert technical ATS parser. Analyze the following Job Description and extract a balanced list of 15 to 20 key competencies.
    
    EXTRACTION RULES:
    1. Include BOTH broad domain concepts (e.g., 'unit testing', 'rest apis', 'database management', 'agile', 'linux networking') AND specific tools/frameworks (e.g., 'java', 'react', 'aws', 'terraform', 'spring boot').
    2. Normalize all framework/library names to plain terms (e.g., write 'react' instead of 'react.js', 'node' instead of 'node.js').
    3. DO NOT include generic filler verbs ('build', 'maintain', 'collaborate', 'develop', 'support').
    4. DO NOT include company names ('barclays'), locations ('bengaluru'), degrees, or soft skills.
    5. Return ONLY a clean comma-separated list in lowercase.
    
    Job Description:
    {jd_text}
    """
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt
        )
        raw_skills = [s.strip() for s in response.text.split(',') if s.strip()]
        normalized_skills = list(dict.fromkeys([normalize_skill(s) for s in raw_skills]))
        return normalized_skills
    except Exception:
        return []

# Upload Section
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("1. Upload Candidate Resume (PDF)", type=["pdf"])
with col2:
    job_description = st.text_area("2. Paste Job Description (JD)", height=160)

# Calculate and store results in Session State upon button click
if st.button("🚀 Analyze Compatibility", type="primary", use_container_width=True):
    if uploaded_file is not None and job_description.strip() != "":
        with st.spinner("SMARA is analyzing candidate parameters..."):
            
            resume_raw_text = extract_text_from_pdf(uploaded_file)
            
            if len(resume_raw_text.strip()) == 0:
                st.error("⚠️ Unable to extract text from PDF.")
            else:
                resume_text = clean_text(resume_raw_text)
                jd_text = clean_text(job_description)
                
                # Contextual Vector Similarity
                vectorizer = TfidfVectorizer(
                    stop_words='english',
                    token_pattern=r'(?u)\b[a-zA-Z0-9\+\#\-\/]{2,}\b',
                    ngram_range=(1, 1),
                    sublinear_tf=True
                )
                tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
                raw_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
                
                # Dynamic Domain Competency Matching (Normalized Comparison)
                extracted_competencies = extract_domain_competencies(job_description, gemini_client)
                
                # Normalize resume text identically for matching
                resume_normalized = normalize_skill(resume_raw_text)
                
                matching_skills = []
                missing_skills = []

                for skill in extracted_competencies:
                    clean_skill_target = normalize_skill(skill)
                    pattern = r'\b' + re.escape(clean_skill_target) + r'\b'
                    
                    if re.search(pattern, resume_normalized) or clean_skill_target in resume_normalized:
                        matching_skills.append(skill)
                    else:
                        missing_skills.append(skill)
                
                skill_match_score = (len(matching_skills) / len(extracted_competencies) * 100) if extracted_competencies else raw_similarity
                final_ats_score = round((0.40 * raw_similarity) + (0.60 * skill_match_score), 1)
                
                # Color and Status Formatting
                if final_ats_score >= 70:
                    score_color = "#2ECC71"
                    status_text = "🟢 Strong Match — Ready to Apply"
                    badge_bg = "#1E3A2B"
                elif final_ats_score >= 45:
                    score_color = "#F39C12"
                    status_text = "🟡 Moderate Match — Needs Minor Adjustments"
                    badge_bg = "#3A301E"
                else:
                    score_color = "#E74C3C"
                    status_text = "🔴 Needs Tailoring — Significant Skill Gap"
                    badge_bg = "#3A1E1E"

                # Generate Breakdown Chart Image Buffer for Download in Tab 2
                chart_df = pd.DataFrame({
                    'Category': ['Matching Skills', 'Missing Skills'],
                    'Count': [len(matching_skills), len(missing_skills)]
                })
                fig, ax = plt.subplots(figsize=(7, 2.5))
                sns.barplot(data=chart_df, x='Count', y='Category', palette=['#2ECC71', '#E74C3C'], ax=ax)
                ax.set_title("Domain Competency Coverage Ratio", fontsize=11)
                ax.set_xlabel("Number of Skills & Qualifications")
                ax.set_ylabel("")
                
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=300)
                img_buf.seek(0)

                # Generate Text Report for Tab 1
                smara_report_text = f"""SMARA REPORT
===========================
Overall ATS Score: {final_ats_score}%
Status: {status_text}

MATCHING COMPETENCIES ({len(matching_skills)}):
{', '.join(sorted(matching_skills)) if matching_skills else 'None'}

MISSING COMPETENCIES ({len(missing_skills)}):
{', '.join(sorted(missing_skills)) if missing_skills else 'None'}
"""

                # Generate SMARA AI Report if authenticated
                ai_report_text = None
                ai_recommendation_text = None
                
                if api_valid and gemini_client:
                    try:
                        prompt = f"""
                        You are SMARA AI, an executive ATS Resume Coach. The candidate scored {final_ats_score}% for this role.
                        Missing Domain Skills: {', '.join(missing_skills)}
                        Job Description: {job_description}
                        Candidate Resume: {resume_raw_text}

                        Provide 3 strategic bullet-point recommendations to seamlessly incorporate the missing technical skills into the candidate's existing accomplishments.
                        CRITICAL RULES:
                        - Focus ONLY on hard skills, tools, and quantifiable metrics.
                        - NEVER recommend adding the hiring company's name or target location to the resume.
                        - NEVER recommend lying about a skill the candidate clearly lacks; instead, suggest how they can pivot existing adjacent experience to match it.
                        """

                        response = gemini_client.models.generate_content(
                            model='gemini-3.5-flash-lite',
                            contents=prompt
                        )
                        ai_recommendation_text = response.text
                        ai_report_text = f"""===================================
SMARA AI REPORT
===================================
Candidate ATS Score: {final_ats_score}%
Status: {status_text}

MISSING COMPETENCIES:
{', '.join(sorted(missing_skills)) if missing_skills else 'None'}

SMARA AI STRATEGIC RECOMMENDATIONS:
-----------------------------------
{ai_recommendation_text}
"""
                    except Exception as ai_exec_err:
                        ai_recommendation_text = f"Error generating recommendations: {ai_exec_err}"

                # Store all calculation variables in Session State
                st.session_state['results'] = {
                    'final_ats_score': final_ats_score,
                    'skill_match_score': skill_match_score,
                    'raw_similarity': raw_similarity,
                    'matching_skills': matching_skills,
                    'missing_skills': missing_skills,
                    'score_color': score_color,
                    'status_text': status_text,
                    'badge_bg': badge_bg,
                    'fig': fig,
                    'graph_png_bytes': img_buf.getvalue(),
                    'smara_report_text': smara_report_text,
                    'ai_report_text': ai_report_text,
                    'ai_recommendation_text': ai_recommendation_text
                }
    else:
        st.error("Please upload a PDF resume and provide a target job description.")

# Render UI from Session State to prevent download buttons from wiping data
if 'results' in st.session_state:
    res = st.session_state['results']
    
    st.divider()

    # Hero Score Header
    st.markdown(f"""
    <div class="hero-card">
        <div style="color: #A6ADC8; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">SMARA Overall ATS Match Score</div>
        <div class="hero-score" style="color: {res['score_color']};">{res['final_ats_score']}%</div>
        <div class="status-badge" style="background-color: {res['badge_bg']}; color: {res['score_color']}; border: 1px solid {res['score_color']};">
            {res['status_text']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sub-Score Breakdown Expander
    with st.expander("🔍 View Detailed Score Breakdown"):
        sc1, sc2 = st.columns(2)
        sc1.metric("Domain Competency Match Rate", f"{round(res['skill_match_score'], 1)}%", help="Percentage of core required domain skills present on candidate resume.")
        sc2.metric("Contextual Vector Score", f"{round(res['raw_similarity'], 1)}%", help="TF-IDF keyword density similarity.")

    # Tabbed Interface
    tab1, tab2, tab3 = st.tabs(["🛠️ Skill Gap Analysis", "📈 Breakdown Chart", "🤖 SMARA AI Coach"])

    # TAB 1: Skill Gap Analysis & SMARA Report Download
    with tab1:
        t_col1, t_col2 = st.columns(2)
        
        with t_col1:
            st.markdown(f"#### 🎯 Matching Skills & Qualifications ({len(res['matching_skills'])})")
            if res['matching_skills']:
                badges_html = "".join([f'<span class="badge-matched">✓ {skill.upper()}</span>' for skill in sorted(res['matching_skills'])])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.info("No direct skill matches detected.")

        with t_col2:
            st.markdown(f"#### ⚠️ Missing Skills & Qualifications ({len(res['missing_skills'])})")
            if res['missing_skills']:
                badges_html = "".join([f'<span class="badge-missing">✕ {skill.upper()}</span>' for skill in sorted(res['missing_skills'])])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.success("All core required domain skills are present!")

        st.divider()
        st.download_button(
            label="📥 Download SMARA Report (.txt)",
            data=res['smara_report_text'],
            file_name="SMARA Report.txt",
            mime="text/plain",
            use_container_width=True
        )

    # TAB 2: Breakdown Chart & SMARA Graph Download
    with tab2:
        st.pyplot(res['fig'])
        st.divider()
        st.download_button(
            label="📥 Download SMARA Graph (.png)",
            data=res['graph_png_bytes'],
            file_name="SMARA Graph.png",
            mime="image/png",
            use_container_width=True
        )

    # TAB 3: SMARA AI Coach & SMARA AI Report Download
    with tab3:
        st.markdown("### 🤖 SMARA AI Report")
        if api_valid and res['ai_recommendation_text']:
            st.markdown(res['ai_recommendation_text'])
            st.divider()
            if res['ai_report_text']:
                st.download_button(
                    label="📥 Download SMARA AI Report (.txt)",
                    data=res['ai_report_text'],
                    file_name="SMARA AI Report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            if not user_api_key:
                st.info("💡 API Key missing. Add `GEMINI_API_KEY` to `.streamlit/secrets.toml` or enter it in the sidebar to generate the SMARA AI Report.")
            else:
                st.error("❌ Authentication failed for Gemini API Key. Please verify key credentials in the sidebar.")