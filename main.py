"""
Snowflake Certification Prep Application
A comprehensive study tool for SnowPro certifications
"""
import streamlit as st
import json
from snowflake_utils import (
    check_connection_status,
    generate_quiz_with_cortex,
    generate_sql_lab,
    generate_mock_quiz,
    generate_mock_sql_lab
)

st.set_page_config(
    page_title="Snowflake Certification Prep",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        background: linear-gradient(90deg, #29b5e8 0%, #1a73e8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #a3a8b4;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .domain-card {
        background-color: #1a1f2e;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #29b5e8;
    }
    .weight-badge {
        background-color: #29b5e8;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .resource-link {
        color: #29b5e8;
        text-decoration: none;
    }
    .resource-link:hover {
        color: #1a73e8;
        text-decoration: underline;
    }
    .score-display {
        background: linear-gradient(135deg, #1a1f2e 0%, #252b3d 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #29b5e8;
    }
    .correct-answer {
        background-color: #1a3d1a;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .incorrect-answer {
        background-color: #3d1a1a;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .quiz-option {
        background-color: #1a1f2e;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .quiz-option:hover {
        background-color: #252b3d;
    }
    div[data-testid="stExpander"] {
        background-color: #1a1f2e;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_exam_data():
    """Load exam configuration data."""
    with open('exam_data.json', 'r') as f:
        return json.load(f)


def init_session_state():
    """Initialize session state variables."""
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'total_questions' not in st.session_state:
        st.session_state.total_questions = 0
    if 'current_quiz' not in st.session_state:
        st.session_state.current_quiz = None
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}


def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.markdown("## ❄️ Snowflake Cert Prep")
    st.sidebar.markdown("---")
    
    exam_data = load_exam_data()
    exams = exam_data['exams']
    
    exam_options = {code: f"{data['name']} ({code})" for code, data in exams.items()}
    
    selected_exam = st.sidebar.selectbox(
        "Select Certification Exam",
        options=list(exam_options.keys()),
        format_func=lambda x: exam_options[x],
        index=3
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Session Progress")
    
    if st.session_state.total_questions > 0:
        accuracy = (st.session_state.score / st.session_state.total_questions) * 100
        st.sidebar.metric("Score", f"{st.session_state.score}/{st.session_state.total_questions}")
        st.sidebar.progress(accuracy / 100)
        st.sidebar.caption(f"Accuracy: {accuracy:.1f}%")
    else:
        st.sidebar.info("Complete quizzes to track progress")
    
    if st.sidebar.button("🔄 Reset Score"):
        st.session_state.score = 0
        st.session_state.total_questions = 0
        st.rerun()
    
    st.sidebar.markdown("---")
    conn_status = check_connection_status()
    if conn_status['configured']:
        st.sidebar.success("✅ Snowflake Connected")
    else:
        st.sidebar.warning("⚠️ Demo Mode (No Snowflake)")
        st.sidebar.caption("Configure Snowflake credentials for AI-powered features")
    
    return selected_exam


def render_roadmap(exam_code: str, exam_data: dict):
    """Render the dynamic roadmap for the selected exam."""
    exam = exam_data['exams'][exam_code]
    
    st.markdown(f"### 🗺️ {exam['name']} Roadmap")
    st.markdown(f"*{exam['description']}*")
    st.markdown("---")
    
    cols = st.columns(2)
    for idx, domain in enumerate(exam['domains']):
        with cols[idx % 2]:
            with st.container():
                st.markdown(f"""
                <div class="domain-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <strong style="color: #e0e0e0; font-size: 1.1rem;">{domain['name']}</strong>
                        <span class="weight-badge">{domain['weight']}%</span>
                    </div>
                    <div style="background-color: #252b3d; border-radius: 10px; height: 8px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #29b5e8, #1a73e8); width: {domain['weight']}%; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📚 View Topics"):
                    for topic in domain['topics']:
                        st.markdown(f"• {topic}")


def render_mindmap(exam_code: str, exam_data: dict):
    """Render mind maps for exam domains."""
    try:
        from streamlit_mermaid import st_mermaid
        has_mermaid = True
    except ImportError:
        has_mermaid = False
    
    exam = exam_data['exams'][exam_code]
    mindmaps = exam_data.get('mindmaps', {}).get(exam_code, {})
    
    st.markdown(f"### 🧠 {exam['name']} Mind Maps")
    st.markdown("*Visual concept maps for each domain*")
    st.markdown("---")
    
    domain_names = [d['name'] for d in exam['domains']]
    selected_domain = st.selectbox("Select Domain", domain_names)
    
    if selected_domain in mindmaps:
        mermaid_code = mindmaps[selected_domain]
        
        if has_mermaid:
            st_mermaid(mermaid_code, height="400")
        else:
            st.code(mermaid_code, language="mermaid")
            st.info("Install streamlit-mermaid for interactive diagrams: `pip install streamlit-mermaid`")
    else:
        st.info("Mind map not available for this domain yet.")


def render_quiz(exam_code: str, exam_data: dict):
    """Render the AI-powered quiz engine."""
    exam = exam_data['exams'][exam_code]
    
    st.markdown(f"### 📝 {exam['name']} Quiz")
    st.markdown("*Test your knowledge with scenario-based questions*")
    st.markdown("---")
    
    domain_names = [d['name'] for d in exam['domains']]
    all_topics = []
    for domain in exam['domains']:
        all_topics.extend(domain['topics'])
    
    col1, col2 = st.columns(2)
    with col1:
        selected_domain = st.selectbox("Select Domain", domain_names, key="quiz_domain")
    with col2:
        domain_topics = next((d['topics'] for d in exam['domains'] if d['name'] == selected_domain), [])
        selected_topic = st.selectbox("Select Topic", ["All Topics"] + domain_topics, key="quiz_topic")
    
    topic_for_quiz = selected_domain if selected_topic == "All Topics" else selected_topic
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        generate_btn = st.button("🎯 Generate Quiz", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 New Quiz", use_container_width=True):
            st.session_state.current_quiz = None
            st.session_state.quiz_submitted = False
            st.session_state.user_answers = {}
            st.rerun()
    
    if generate_btn:
        with st.spinner("Generating quiz questions..."):
            conn_status = check_connection_status()
            if conn_status['configured']:
                quiz_response = generate_quiz_with_cortex(exam['name'], topic_for_quiz)
                if quiz_response:
                    try:
                        st.session_state.current_quiz = json.loads(quiz_response)
                    except json.JSONDecodeError:
                        st.session_state.current_quiz = generate_mock_quiz(exam_code, topic_for_quiz)
                else:
                    st.session_state.current_quiz = generate_mock_quiz(exam_code, topic_for_quiz)
            else:
                st.session_state.current_quiz = generate_mock_quiz(exam_code, topic_for_quiz)
            
            st.session_state.quiz_submitted = False
            st.session_state.user_answers = {}
            st.rerun()
    
    if st.session_state.current_quiz:
        st.markdown("---")
        questions = st.session_state.current_quiz
        
        for i, q in enumerate(questions):
            st.markdown(f"**Question {i+1}:** {q['question']}")
            
            options = q['options']
            option_labels = [opt.split(') ', 1)[-1] if ') ' in opt else opt for opt in options]
            option_keys = ['A', 'B', 'C', 'D'][:len(options)]
            
            if st.session_state.quiz_submitted:
                user_answer = st.session_state.user_answers.get(i)
                correct_answer = q['correct_answer']
                
                for j, (key, label) in enumerate(zip(option_keys, option_labels)):
                    if key == correct_answer:
                        st.markdown(f"""<div class="correct-answer">✅ {key}) {label}</div>""", unsafe_allow_html=True)
                    elif key == user_answer and user_answer != correct_answer:
                        st.markdown(f"""<div class="incorrect-answer">❌ {key}) {label}</div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"⬜ {key}) {label}")
                
                with st.expander("💡 View Explanation"):
                    st.markdown(q.get('explanation', 'No explanation available.'))
            else:
                selected = st.radio(
                    f"Select answer for Q{i+1}",
                    options=option_keys,
                    format_func=lambda x, opts=options, keys=option_keys: opts[keys.index(x)] if x in keys else x,
                    key=f"q_{i}",
                    horizontal=True,
                    label_visibility="collapsed"
                )
                st.session_state.user_answers[i] = selected
            
            st.markdown("---")
        
        if not st.session_state.quiz_submitted:
            if st.button("✅ Submit Quiz", type="primary", use_container_width=True):
                correct_count = 0
                for i, q in enumerate(questions):
                    if st.session_state.user_answers.get(i) == q['correct_answer']:
                        correct_count += 1
                
                st.session_state.score += correct_count
                st.session_state.total_questions += len(questions)
                st.session_state.quiz_submitted = True
                st.rerun()
        else:
            correct_count = sum(1 for i, q in enumerate(questions) 
                              if st.session_state.user_answers.get(i) == q['correct_answer'])
            
            st.markdown(f"""
            <div class="score-display">
                <h2 style="color: #29b5e8; margin-bottom: 0.5rem;">Quiz Complete!</h2>
                <h1 style="color: white; margin: 0;">{correct_count}/{len(questions)}</h1>
                <p style="color: #a3a8b4;">Questions Correct</p>
            </div>
            """, unsafe_allow_html=True)


def render_sql_lab(exam_code: str, exam_data: dict):
    """Render the SQL Lab generator."""
    exam = exam_data['exams'][exam_code]
    
    st.markdown(f"### 💻 SQL Practice Lab")
    st.markdown("*Generate copy-pasteable SQL scripts for hands-on practice*")
    st.markdown("---")
    
    lab_features = [
        "Vector Search",
        "Streams and Tasks",
        "Dynamic Tables",
        "Cortex LLM Functions",
        "Data Masking",
        "Snowpipe Setup",
        "Time Travel Queries",
        "Secure Views",
        "Row Access Policies",
        "External Tables"
    ]
    
    if exam_code == "GES-C01":
        lab_features = [
            "Vector Search",
            "Cortex LLM Functions",
            "EMBED_TEXT Function",
            "RAG Implementation",
            "Document Processing",
            "Sentiment Analysis",
            "Text Summarization",
            "Model Fine-tuning"
        ]
    elif exam_code == "DEA-C01":
        lab_features = [
            "Streams and Tasks",
            "Dynamic Tables",
            "Snowpipe Setup",
            "Task DAGs",
            "Error Handling in Procedures",
            "Data Quality Checks",
            "CDC Pipelines",
            "Snowpark Python UDFs"
        ]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_feature = st.selectbox("Select Feature to Practice", lab_features)
    with col2:
        custom_feature = st.text_input("Or enter custom feature")
    
    feature_to_generate = custom_feature if custom_feature else selected_feature
    
    if st.button("🔧 Generate SQL Lab", type="primary"):
        with st.spinner("Generating SQL practice scripts..."):
            conn_status = check_connection_status()
            if conn_status['configured']:
                sql_lab = generate_sql_lab(feature_to_generate)
                if not sql_lab:
                    sql_lab = generate_mock_sql_lab(feature_to_generate)
            else:
                sql_lab = generate_mock_sql_lab(feature_to_generate)
            
            st.session_state.current_sql_lab = sql_lab
    
    if 'current_sql_lab' in st.session_state and st.session_state.current_sql_lab:
        st.markdown("### Generated SQL Script")
        st.code(st.session_state.current_sql_lab, language="sql")
        
        st.download_button(
            "📥 Download SQL Script",
            st.session_state.current_sql_lab,
            file_name=f"{feature_to_generate.lower().replace(' ', '_')}_lab.sql",
            mime="text/plain"
        )


def render_resources(exam_code: str, exam_data: dict):
    """Render the reference library."""
    exam = exam_data['exams'][exam_code]
    
    st.markdown(f"### 📚 Reference Library")
    st.markdown(f"*Curated resources for {exam['name']}*")
    st.markdown("---")
    
    st.markdown("#### 🔗 Official Documentation")
    for resource in exam.get('resources', []):
        st.markdown(f"• [{resource['title']}]({resource['url']})")
    
    st.markdown("---")
    st.markdown("#### 📖 Study Tips")
    
    tips = {
        "COF-C02": [
            "Focus on understanding the three-layer architecture (Cloud Services, Compute, Storage)",
            "Practice COPY INTO commands and understand file format options",
            "Master role-based access control (RBAC) concepts",
            "Understand Time Travel and Fail-safe retention periods",
            "Know the differences between Snowflake editions"
        ],
        "DEA-C01": [
            "Deep dive into Streams and Tasks for CDC pipelines",
            "Practice Dynamic Tables for declarative data pipelines",
            "Understand Snowpipe vs Snowpipe Streaming use cases",
            "Learn Snowpark Python for complex transformations",
            "Master task DAGs and error handling patterns"
        ],
        "ARA-C01": [
            "Understand multi-account and organization strategies",
            "Study database replication and failover patterns",
            "Learn private connectivity options (PrivateLink, etc.)",
            "Master cost optimization and workload management",
            "Know data sharing and marketplace strategies"
        ],
        "GES-C01": [
            "Master Cortex LLM functions: COMPLETE, SUMMARIZE, TRANSLATE",
            "Understand vector embeddings and EMBED_TEXT function",
            "Learn RAG (Retrieval Augmented Generation) architecture",
            "Practice vector similarity search with VECTOR type",
            "Study Document AI capabilities and use cases",
            "Understand security considerations for AI/ML workloads"
        ]
    }
    
    for tip in tips.get(exam_code, []):
        st.markdown(f"✅ {tip}")


def main():
    """Main application entry point."""
    init_session_state()
    
    selected_exam = render_sidebar()
    exam_data = load_exam_data()
    exam = exam_data['exams'][selected_exam]
    
    st.markdown(f'<h1 class="main-header">❄️ Snowflake Certification Prep</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Prepare for your {exam["name"]} certification</p>', unsafe_allow_html=True)
    
    tab_roadmap, tab_mindmap, tab_quiz, tab_lab, tab_resources = st.tabs([
        "🗺️ Roadmap", 
        "🧠 Mind Maps", 
        "📝 Quiz", 
        "💻 SQL Lab", 
        "📚 Resources"
    ])
    
    with tab_roadmap:
        render_roadmap(selected_exam, exam_data)
    
    with tab_mindmap:
        render_mindmap(selected_exam, exam_data)
    
    with tab_quiz:
        render_quiz(selected_exam, exam_data)
    
    with tab_lab:
        render_sql_lab(selected_exam, exam_data)
    
    with tab_resources:
        render_resources(selected_exam, exam_data)


if __name__ == "__main__":
    main()
