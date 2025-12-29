"""
Snowflake utility functions for Cortex AI and database connections.
Uses latest Cortex AI syntax and best practices.
"""
import os
import json
import streamlit as st
import snowflake.connector
from typing import Optional, Dict, Any, List


@st.cache_resource
def get_snowflake_connection():
    """
    Get Snowflake connection using snowflake-connector-python.
    Credentials are read from environment variables.
    """
    try:
        conn = snowflake.connector.connect(
            account=os.environ.get('SNOWFLAKE_ACCOUNT'),
            user=os.environ.get('SNOWFLAKE_USER'),
            password=os.environ.get('SNOWFLAKE_PASSWORD'),
            warehouse=os.environ.get('SNOWFLAKE_WAREHOUSE'),
            database=os.environ.get('SNOWFLAKE_DATABASE', 'SNOWFLAKE'),
            schema=os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC'),
            role=os.environ.get('SNOWFLAKE_ROLE', 'ACCOUNTADMIN')
        )
        return conn
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {str(e)}")
        return None


def check_connection_status() -> Dict[str, Any]:
    """
    Check if Snowflake connection is properly configured.
    Returns status dict with connection info.
    """
    required_vars = ['SNOWFLAKE_ACCOUNT', 'SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD', 
                     'SNOWFLAKE_WAREHOUSE']
    
    status = {
        "configured": False,
        "missing_vars": [],
        "message": ""
    }
    
    for var in required_vars:
        if not os.environ.get(var):
            status["missing_vars"].append(var)
    
    if not status["missing_vars"]:
        status["configured"] = True
        status["message"] = "Snowflake connection is configured"
    else:
        status["message"] = f"Missing configuration: {', '.join(status['missing_vars'])}"
    
    return status


def escape_sql_string(s: str) -> str:
    """Escape single quotes for SQL strings."""
    return s.replace("'", "''").replace("\\", "\\\\")


def generate_quiz_with_cortex(exam_name: str, topic: str, model: str = "mistral-large") -> Optional[str]:
    """
    Generate quiz questions using Snowflake Cortex AI_COMPLETE function.
    Uses latest Cortex syntax with proper prompt engineering.
    """
    system_prompt = f"""You are a senior Snowflake certification instructor with deep expertise in {exam_name}. 
Your task is to create 5 challenging, exam-quality multiple-choice questions on: {topic}

REQUIREMENTS:
1. Questions must be scenario-based reflecting real-world Snowflake usage
2. Include practical SQL examples or architecture decisions where applicable
3. Distractors should be plausible but clearly wrong to someone who understands the concept
4. Focus on "why" and "when" not just "what"
5. Reference actual Snowflake features, functions, and best practices from official documentation

EXAM-SPECIFIC FOCUS:
- COF-C02: Focus on core concepts, architecture layers, RBAC, Time Travel, editions
- DEA-C01: Focus on Streams, Tasks, Dynamic Tables, Snowpipe, Snowpark
- ARA-C01: Focus on multi-account, replication, private connectivity, cost optimization
- GES-C01: Focus on Cortex LLM functions, Cortex Search, Cortex Analyst, Document AI, Cortex Guard

RESPONSE FORMAT - Return ONLY valid JSON array:
[
  {{
    "question": "Scenario-based question text",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct_answer": "A",
    "explanation": "Detailed explanation with Snowflake documentation reference"
  }}
]"""

    user_prompt = f"Generate 5 exam-quality questions for {exam_name} certification on the topic: {topic}. Return ONLY the JSON array."

    conn = get_snowflake_connection()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        escaped_system = escape_sql_string(system_prompt)
        escaped_user = escape_sql_string(user_prompt)
        
        query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            ARRAY_CONSTRUCT(
                OBJECT_CONSTRUCT('role', 'system', 'content', '{escaped_system}'),
                OBJECT_CONSTRUCT('role', 'user', 'content', '{escaped_user}')
            ),
            OBJECT_CONSTRUCT('temperature', 0.3, 'max_tokens', 4000)
        ) as response
        """
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        
        if result and result[0]:
            response = result[0]
            if isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, dict) and 'choices' in parsed:
                        content = parsed['choices'][0]['messages']
                        return content
                    return response
                except json.JSONDecodeError:
                    return response
            return str(response)
        return None
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        return None


def generate_sql_lab(feature: str, model: str = "mistral-large") -> Optional[str]:
    """
    Generate production-quality SQL practice scripts using Snowflake Cortex.
    Uses detailed prompts based on official Snowflake documentation patterns.
    """
    
    feature_prompts = get_feature_specific_prompt(feature)
    
    system_prompt = f"""You are a Snowflake Solutions Architect creating production-grade SQL lab exercises.
Create a comprehensive, educational SQL lab for: {feature}

{feature_prompts}

CRITICAL REQUIREMENTS:
1. Use ONLY official Snowflake syntax from latest documentation
2. Include realistic sample data that demonstrates the feature properly
3. Add detailed comments explaining each step and why it matters
4. Show common use cases and best practices
5. Include error handling where appropriate
6. Add monitoring/verification queries
7. Include proper cleanup commands (commented out)

STRUCTURE YOUR LAB AS:
-- ============================================
-- SQL Lab: [Feature Name]
-- Based on Snowflake Documentation Best Practices
-- ============================================

-- SECTION 1: Environment Setup
-- SECTION 2: Create Required Objects  
-- SECTION 3: Insert Sample Data
-- SECTION 4: Core Feature Demonstration
-- SECTION 5: Advanced Usage Examples
-- SECTION 6: Monitoring and Verification
-- SECTION 7: Cleanup (commented)

Make the lab copy-pasteable and executable in Snowsight."""

    user_prompt = f"Create the complete SQL lab for {feature} following all requirements. Output only SQL code with comments."

    conn = get_snowflake_connection()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        escaped_system = escape_sql_string(system_prompt)
        escaped_user = escape_sql_string(user_prompt)
        
        query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            ARRAY_CONSTRUCT(
                OBJECT_CONSTRUCT('role', 'system', 'content', '{escaped_system}'),
                OBJECT_CONSTRUCT('role', 'user', 'content', '{escaped_user}')
            ),
            OBJECT_CONSTRUCT('temperature', 0.2, 'max_tokens', 8000)
        ) as response
        """
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        
        if result and result[0]:
            response = result[0]
            if isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, dict) and 'choices' in parsed:
                        content = parsed['choices'][0]['messages']
                        return content
                    return response
                except json.JSONDecodeError:
                    return response
            return str(response)
        return None
    except Exception as e:
        st.error(f"Error generating SQL lab: {str(e)}")
        return None


def get_feature_specific_prompt(feature: str) -> str:
    """Get feature-specific documentation context for better lab generation."""
    
    prompts = {
        "Cortex LLM Functions (COMPLETE, SUMMARIZE, TRANSLATE)": """
SNOWFLAKE CORTEX LLM FUNCTIONS REFERENCE:
- SNOWFLAKE.CORTEX.COMPLETE(model, prompt) - Generate text using LLMs
- SNOWFLAKE.CORTEX.COMPLETE(model, messages_array, options_object) - Chat format with system/user roles
- SNOWFLAKE.CORTEX.SUMMARIZE(text) - Summarize long text
- SNOWFLAKE.CORTEX.SENTIMENT(text) - Returns sentiment score -1 to 1
- SNOWFLAKE.CORTEX.TRANSLATE(text, from_lang, to_lang) - Translate text
- SNOWFLAKE.CORTEX.EXTRACT_ANSWER(context, question) - Extract answers from context

Available models: mistral-large, mistral-7b, llama3-70b, llama3-8b, llama3.1-70b
Options: temperature (0-1), max_tokens, top_p

Show examples of each function with realistic business scenarios.""",

        "Cortex Search Setup": """
CORTEX SEARCH SERVICE REFERENCE:
- CREATE CORTEX SEARCH SERVICE service_name
  ON table_name
  ATTRIBUTES column1, column2
  WAREHOUSE = warehouse_name
  TARGET_LAG = 'time_interval';

- Use for RAG (Retrieval Augmented Generation) applications
- Automatically creates and maintains vector embeddings
- Supports hybrid search (semantic + keyword)
- Query using SEARCH() function or REST API

Show complete setup with document table, search service creation, and query examples.""",

        "Cortex Analyst Semantic Model": """
CORTEX ANALYST REFERENCE:
- Enables natural language to SQL (Text-to-SQL)
- Requires semantic model YAML definition
- Semantic model defines tables, columns, relationships, metrics
- Query via REST API or Streamlit integration

YAML Structure:
- name, tables, dimensions, measures, time_dimensions
- Verified queries for common questions

Show semantic model YAML creation and query examples.""",

        "Cortex Agent Multi-turn Chat": """
CORTEX AGENT REFERENCE:
- Build conversational AI with tool calling capabilities
- Multi-turn conversation with context retention
- Integrate with Cortex Search for RAG
- Support for custom tools and functions

Architecture:
- Session management with conversation history
- Tool definitions and execution
- Response streaming support

Show chat implementation with session state and tool integration.""",

        "Vector Embeddings with EMBED_TEXT": """
VECTOR EMBEDDINGS REFERENCE:
- SNOWFLAKE.CORTEX.EMBED_TEXT_768(model, text) - Returns 768-dim vector
- SNOWFLAKE.CORTEX.EMBED_TEXT_1024(model, text) - Returns 1024-dim vector
- Models: e5-base-v2, snowflake-arctic-embed-m

VECTOR functions:
- VECTOR_COSINE_SIMILARITY(v1, v2) - Cosine similarity
- VECTOR_L2_DISTANCE(v1, v2) - Euclidean distance
- VECTOR_INNER_PRODUCT(v1, v2) - Dot product

Show document embedding storage and semantic search implementation.""",

        "RAG Implementation": """
RAG (RETRIEVAL AUGMENTED GENERATION) REFERENCE:
1. Document ingestion and chunking
2. Generate embeddings using EMBED_TEXT
3. Store in table with VECTOR column
4. Query: embed question, find similar chunks
5. Pass context to COMPLETE for grounded response

Best practices:
- Chunk size 500-1000 tokens with overlap
- Use metadata filtering
- Combine with Cortex Search for production

Show complete RAG pipeline from document to answer.""",

        "AI_PARSE_DOCUMENT for Unstructured Data": """
AI_PARSE_DOCUMENT REFERENCE:
- SNOWFLAKE.CORTEX.PARSE_DOCUMENT(@stage/file.pdf, options)
- Extracts text, tables, and structure from documents
- Supports PDF, images, Word documents

Options: 
- mode: 'LAYOUT' (preserve formatting) or 'OCR' (extract all text)

Pipeline:
1. Upload documents to stage
2. Parse with PARSE_DOCUMENT
3. Store extracted content
4. Process with LLM functions

Show document processing pipeline with stage setup.""",

        "Cortex Fine Tuning": """
CORTEX FINE-TUNING REFERENCE:
- CREATE CORTEX FINE_TUNING_JOB job_name
  BASE_MODEL = 'model_name'
  TRAINING_DATA = table_name
  VALIDATION_DATA = table_name;

Training data format:
- prompt, completion columns
- Or messages array for chat format

Show fine-tuning job creation, monitoring, and model usage.""",

        "Cortex Guard Implementation": """
CORTEX GUARD REFERENCE:
- SNOWFLAKE.CORTEX.GUARD(model, prompt, guardrails)
- Filters harmful/inappropriate content
- Categories: hate, violence, sexual, self-harm, etc.

Usage patterns:
- Pre-check user inputs before LLM
- Post-check LLM outputs before display
- Configure sensitivity thresholds

Show input/output filtering with guardrail configuration.""",

        "Document AI Setup and PREDICT Queries": """
DOCUMENT AI REFERENCE:
- Build custom document extraction models
- Train on your document types
- Use !PREDICT for extraction

Setup:
1. CREATE DOCUMENT AI BUILD build_name
2. Upload training documents
3. Define questions/fields to extract
4. Train model
5. Deploy and use !PREDICT

Show complete Document AI workflow with extraction queries.""",

        "TruLens Observability Setup": """
TRULENS SDK FOR AI OBSERVABILITY:
- Track LLM application performance
- Measure: groundedness, relevance, coherence
- Store evaluation results in Snowflake

Setup:
1. Install trulens-snowflake package
2. Configure connection
3. Wrap LLM calls with TruLens
4. Query evaluation metrics

Show observability setup with metric tracking.""",

        "Model Registry with SPCS": """
SNOWFLAKE MODEL REGISTRY & SPCS:
- Register custom ML models
- Deploy models as services via SPCS
- Version control and lineage tracking

Model Registry:
- snowflake.ml.registry.Registry
- log_model(), get_model()

SPCS Deployment:
- CREATE SERVICE with model container
- Scale with compute pools

Show model registration and deployment workflow.""",

        "Streams and Tasks": """
STREAMS AND TASKS REFERENCE:
- CREATE STREAM stream_name ON TABLE table_name
- Captures INSERT, UPDATE, DELETE (CDC)
- METADATA$ACTION, METADATA$ISUPDATE columns

TASKS:
- CREATE TASK task_name WAREHOUSE=wh SCHEDULE='interval'
- WHEN SYSTEM$STREAM_HAS_DATA('stream')
- Task DAGs with AFTER clause

Show CDC pipeline with stream processing and task orchestration.""",

        "Dynamic Tables": """
DYNAMIC TABLES REFERENCE:
- CREATE DYNAMIC TABLE table_name
  TARGET_LAG = 'interval'
  WAREHOUSE = warehouse_name
  AS SELECT query;

- Declarative data pipelines
- Automatic refresh based on target lag
- Monitor with DYNAMIC_TABLE_REFRESH_HISTORY()

Show pipeline with staging, transformation, and aggregation layers.""",
    }
    
    return prompts.get(feature, f"Create a comprehensive lab for {feature} using official Snowflake syntax and best practices.")


def generate_mock_quiz(exam_name: str, topic: str) -> List[Dict]:
    """
    Generate mock quiz questions for demo/offline mode.
    Used when Snowflake connection is not available.
    """
    mock_questions = {
        "COF-C02": {
            "default": [
                {
                    "question": "A company needs to ensure that a specific virtual warehouse automatically scales during peak hours but maintains cost control. Which configuration achieves this?",
                    "options": ["A) Set MIN_CLUSTER_COUNT=1 and MAX_CLUSTER_COUNT=10 with AUTO_SUSPEND=60", "B) Use a larger warehouse size instead of multi-cluster", "C) Disable auto-suspend to keep warehouse running", "D) Set SCALING_POLICY to ECONOMY and MAX_CLUSTER_COUNT=1"],
                    "correct_answer": "A",
                    "explanation": "Multi-cluster warehouses with MIN/MAX cluster settings and AUTO_SUSPEND provide automatic scaling during peak loads while suspending during idle periods for cost control. ECONOMY scaling policy adds clusters more conservatively."
                },
                {
                    "question": "Which Snowflake layer is responsible for query compilation, optimization, and access control decisions?",
                    "options": ["A) Storage Layer", "B) Compute Layer (Virtual Warehouses)", "C) Cloud Services Layer", "D) Network Layer"],
                    "correct_answer": "C",
                    "explanation": "The Cloud Services Layer handles query parsing, compilation, optimization, metadata management, authentication, and access control. This layer runs 24/7 and is shared across all accounts."
                },
                {
                    "question": "A user accidentally deleted a critical table 3 days ago. The account is on Enterprise Edition with default settings. What recovery options are available?",
                    "options": ["A) Use UNDROP TABLE to restore it", "B) Contact Snowflake support for Fail-safe recovery", "C) Both A and B are possible", "D) The data is permanently lost"],
                    "correct_answer": "C",
                    "explanation": "Enterprise Edition has 90-day Time Travel (default 1 day). If within Time Travel period, use UNDROP. If past Time Travel but within 7-day Fail-safe period, contact Snowflake support for recovery."
                },
                {
                    "question": "What is the primary difference between Standard and Enterprise editions regarding data protection?",
                    "options": ["A) Enterprise has encryption, Standard does not", "B) Enterprise supports up to 90 days Time Travel, Standard up to 1 day", "C) Standard has no Fail-safe protection", "D) Enterprise has automatic backups, Standard requires manual"],
                    "correct_answer": "B",
                    "explanation": "Both editions have encryption and 7-day Fail-safe. The key difference is Time Travel: Standard supports 0-1 days, Enterprise supports 0-90 days. This affects point-in-time recovery capabilities."
                },
                {
                    "question": "A data provider wants to share live data with a consumer without data copying. Which approach should they use?",
                    "options": ["A) Export data to cloud storage and share access", "B) Create a Secure Share with read-only access", "C) Set up replication to consumer account", "D) Use Snowpipe to load data to consumer"],
                    "correct_answer": "B",
                    "explanation": "Secure Data Sharing provides zero-copy data sharing where consumers access provider's data directly through secure views. No data movement or copying occurs, and consumers see real-time data."
                }
            ]
        },
        "DEA-C01": {
            "default": [
                {
                    "question": "A data engineer needs to capture all changes (inserts, updates, deletes) from a source table for incremental processing. Which combination should they use?",
                    "options": ["A) Materialized View with scheduled refresh", "B) Stream on source table with Task for processing", "C) Dynamic Table with 1-minute target lag", "D) Snowpipe with COPY INTO"],
                    "correct_answer": "B",
                    "explanation": "Streams capture CDC (Change Data Capture) including all DML operations with METADATA$ACTION and METADATA$ISUPDATE columns. Tasks can be triggered WHEN SYSTEM$STREAM_HAS_DATA() for efficient processing."
                },
                {
                    "question": "What is the key advantage of Dynamic Tables over traditional Streams and Tasks for data pipelines?",
                    "options": ["A) Lower cost per query", "B) Declarative definition with automatic refresh management", "C) Faster query performance", "D) Support for more data types"],
                    "correct_answer": "B",
                    "explanation": "Dynamic Tables use declarative SQL to define transformations, and Snowflake automatically manages the refresh pipeline. This simplifies pipeline development compared to imperative Streams/Tasks orchestration."
                },
                {
                    "question": "A pipeline needs to process JSON files arriving continuously in cloud storage with sub-minute latency. Which loading method is most appropriate?",
                    "options": ["A) Scheduled COPY INTO every minute", "B) Snowpipe with auto-ingest", "C) Snowpipe Streaming API", "D) External table with AUTO_REFRESH"],
                    "correct_answer": "C",
                    "explanation": "Snowpipe Streaming API provides the lowest latency (seconds) for continuous data loading. Traditional Snowpipe has ~1 minute latency. COPY INTO is batch-oriented. External tables don't ingest data."
                },
                {
                    "question": "Which Snowpark feature allows executing Python code directly on Snowflake compute without moving data?",
                    "options": ["A) Python Worksheets", "B) Stored Procedures and UDFs", "C) External Functions", "D) Python Connector"],
                    "correct_answer": "B",
                    "explanation": "Snowpark Python Stored Procedures and UDFs execute Python code on Snowflake's distributed compute, processing data where it lives. External Functions call external services. Python Connector moves data to client."
                },
                {
                    "question": "A task DAG has tasks A -> B -> C. If Task B fails, what happens to Task C?",
                    "options": ["A) Task C runs anyway with NULL inputs", "B) Task C is skipped and marked as failed", "C) Task C waits indefinitely for B to succeed", "D) Task C runs with data from the previous successful B run"],
                    "correct_answer": "B",
                    "explanation": "In Task DAGs, if a predecessor task fails, dependent tasks are skipped and marked as FAILED_WITH_PREDECESSOR. The failure propagates through the DAG. Error handling requires explicit try/catch in stored procedures."
                }
            ]
        },
        "ARA-C01": {
            "default": [
                {
                    "question": "An organization needs to manage 50 Snowflake accounts across multiple cloud providers with centralized billing and governance. What should they implement?",
                    "options": ["A) Create a master account with data sharing to all others", "B) Set up Snowflake Organization with ORGADMIN role", "C) Use database replication between all accounts", "D) Implement SSO with a single identity provider"],
                    "correct_answer": "B",
                    "explanation": "Snowflake Organizations provide centralized management of multiple accounts with ORGADMIN role for account provisioning, usage monitoring, and centralized billing across cloud providers and regions."
                },
                {
                    "question": "For disaster recovery with RPO < 1 hour and RTO < 1 hour, which architecture should be implemented?",
                    "options": ["A) Database replication with manual failover", "B) Account replication with failover groups", "C) Scheduled data exports to backup storage", "D) Multi-cluster warehouse in primary region"],
                    "correct_answer": "B",
                    "explanation": "Failover Groups with account replication provide automated failover with configurable replication frequency (as low as 1 minute). This achieves low RPO/RTO. Database replication requires manual failover."
                },
                {
                    "question": "A financial company requires all Snowflake traffic to stay on private networks, never traversing public internet. Which solution applies?",
                    "options": ["A) Network policies with IP whitelisting", "B) AWS PrivateLink / Azure Private Link / GCP Private Service Connect", "C) VPN tunnel to Snowflake endpoints", "D) Dedicated Snowflake instance"],
                    "correct_answer": "B",
                    "explanation": "Private connectivity (PrivateLink on AWS, Private Link on Azure, Private Service Connect on GCP) provides private network paths to Snowflake without internet exposure. Network policies filter IPs but traffic still uses internet."
                },
                {
                    "question": "What is the recommended approach for implementing a Data Mesh architecture in Snowflake?",
                    "options": ["A) Single database with schema per domain", "B) Separate accounts per domain with data sharing", "C) One warehouse per domain team", "D) Centralized ETL with domain-specific views"],
                    "correct_answer": "B",
                    "explanation": "Data Mesh principles (domain ownership, data as product) align with separate accounts per domain using Secure Data Sharing for cross-domain access. This provides autonomy while enabling data discovery."
                },
                {
                    "question": "Which feature enables Snowflake to work with open table formats stored in customer-managed cloud storage?",
                    "options": ["A) External tables", "B) Apache Iceberg tables", "C) Hybrid tables", "D) Directory tables"],
                    "correct_answer": "B",
                    "explanation": "Iceberg tables allow Snowflake to read and write Apache Iceberg format in customer cloud storage. This enables interoperability with other engines (Spark, etc.) while maintaining Snowflake performance."
                }
            ]
        },
        "GES-C01": {
            "default": [
                {
                    "question": "Which Cortex function should you use to build a conversational AI application with system prompts and multi-turn conversations?",
                    "options": ["A) SNOWFLAKE.CORTEX.SUMMARIZE()", "B) SNOWFLAKE.CORTEX.COMPLETE() with messages array", "C) SNOWFLAKE.CORTEX.EXTRACT_ANSWER()", "D) SNOWFLAKE.CORTEX.TRANSLATE()"],
                    "correct_answer": "B",
                    "explanation": "COMPLETE() with messages array format supports system prompts and multi-turn conversations: [{role:'system', content:'...'}, {role:'user', content:'...'}, {role:'assistant', content:'...'}]. This enables chat applications with context."
                },
                {
                    "question": "A company wants to enable natural language querying of their sales database. Which Cortex service is specifically designed for this?",
                    "options": ["A) Cortex Search", "B) Cortex Analyst", "C) Cortex Guard", "D) Document AI"],
                    "correct_answer": "B",
                    "explanation": "Cortex Analyst enables Text-to-SQL through semantic models (YAML definitions of tables, relationships, metrics). Users ask questions in natural language and Cortex Analyst generates and executes SQL."
                },
                {
                    "question": "When implementing RAG (Retrieval Augmented Generation) for enterprise documents, which Cortex service provides the most production-ready solution?",
                    "options": ["A) Manual EMBED_TEXT with VECTOR similarity search", "B) Cortex Search Service", "C) Cortex Fine-tuning", "D) PARSE_DOCUMENT only"],
                    "correct_answer": "B",
                    "explanation": "Cortex Search Service automatically handles document chunking, embedding, indexing, and hybrid search (semantic + keyword). It's production-ready with automatic refresh and REST API access."
                },
                {
                    "question": "What is the primary purpose of Cortex Guard in AI applications?",
                    "options": ["A) Authenticate users accessing LLM functions", "B) Filter harmful or inappropriate content in inputs/outputs", "C) Optimize LLM query performance", "D) Encrypt data sent to LLMs"],
                    "correct_answer": "B",
                    "explanation": "Cortex Guard filters content for safety categories (hate, violence, sexual content, etc.) in both user inputs and LLM outputs. It helps prevent harmful content in AI applications."
                },
                {
                    "question": "Which approach should you use to process PDFs and extract structured data using Snowflake Cortex?",
                    "options": ["A) SNOWFLAKE.CORTEX.COMPLETE() with PDF content", "B) SNOWFLAKE.CORTEX.PARSE_DOCUMENT() for extraction", "C) Document AI with custom trained models", "D) Both B and C depending on use case"],
                    "correct_answer": "D",
                    "explanation": "PARSE_DOCUMENT() extracts text/tables from documents with OCR. Document AI allows training custom extraction models for specific document types (invoices, forms). Use PARSE_DOCUMENT for general extraction, Document AI for structured field extraction."
                }
            ]
        }
    }
    
    exam_questions = mock_questions.get(exam_name, mock_questions["COF-C02"])
    return exam_questions.get("default", exam_questions["default"])


def generate_mock_sql_lab(feature: str) -> str:
    """
    Generate production-quality SQL lab scripts for demo/offline mode.
    Based on official Snowflake documentation patterns.
    """
    labs = {
        "Cortex LLM Functions (COMPLETE, SUMMARIZE, TRANSLATE)": """
-- ============================================
-- SQL Lab: Snowflake Cortex LLM Functions
-- Based on Snowflake Documentation Best Practices
-- ============================================

-- SECTION 1: Environment Setup
USE ROLE ACCOUNTADMIN;  -- Or role with CORTEX functions access
USE WAREHOUSE COMPUTE_WH;
CREATE DATABASE IF NOT EXISTS cortex_lab;
USE DATABASE cortex_lab;
CREATE SCHEMA IF NOT EXISTS llm_demo;
USE SCHEMA llm_demo;

-- SECTION 2: Basic COMPLETE Function Usage
-- Simple text generation with single prompt
SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'mistral-large',
    'Explain the concept of micro-partitions in Snowflake in 3 sentences.'
) AS simple_response;

-- SECTION 3: COMPLETE with Chat Format (System + User prompts)
-- This is the recommended format for production applications
SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'llama3.1-70b',
    [
        {'role': 'system', 'content': 'You are a Snowflake Solutions Architect. Provide concise, technical answers.'},
        {'role': 'user', 'content': 'What is the difference between a Standard and Multi-cluster warehouse?'}
    ],
    {'temperature': 0.3, 'max_tokens': 500}
) AS chat_response;

-- SECTION 4: SUMMARIZE Function
-- Summarize long text content
SELECT SNOWFLAKE.CORTEX.SUMMARIZE(
    'Snowflake is a cloud-based data warehousing platform that uses a unique 
    multi-cluster shared data architecture. The architecture separates storage 
    from compute, allowing each layer to scale independently. The storage layer 
    uses cloud object storage (S3, Azure Blob, GCS) to store data in compressed, 
    columnar format called micro-partitions. The compute layer consists of 
    virtual warehouses that are clusters of compute resources. The cloud services 
    layer handles query optimization, transactions, metadata, and security. 
    This separation enables near-unlimited scalability and a pay-per-use model.'
) AS summary;

-- SECTION 5: SENTIMENT Analysis
-- Analyze sentiment in customer feedback
CREATE OR REPLACE TEMPORARY TABLE customer_reviews (
    review_id INT,
    product_name VARCHAR(100),
    review_text TEXT
);

INSERT INTO customer_reviews VALUES
    (1, 'Snowflake Platform', 'Absolutely love Snowflake! The performance is incredible and scaling is seamless.'),
    (2, 'Snowflake Platform', 'Disappointed with the credit consumption. Costs are higher than expected.'),
    (3, 'Snowflake Platform', 'It works okay. Nothing special but gets the job done.'),
    (4, 'Snowflake Platform', 'Game changer for our analytics team! Highly recommend.');

SELECT 
    review_id,
    product_name,
    review_text,
    SNOWFLAKE.CORTEX.SENTIMENT(review_text) AS sentiment_score,
    CASE 
        WHEN SNOWFLAKE.CORTEX.SENTIMENT(review_text) > 0.3 THEN 'Positive'
        WHEN SNOWFLAKE.CORTEX.SENTIMENT(review_text) < -0.3 THEN 'Negative'
        ELSE 'Neutral'
    END AS sentiment_label
FROM customer_reviews;

-- SECTION 6: TRANSLATE Function
SELECT 
    'Welcome to Snowflake Data Cloud' AS original_text,
    SNOWFLAKE.CORTEX.TRANSLATE('Welcome to Snowflake Data Cloud', 'en', 'es') AS spanish,
    SNOWFLAKE.CORTEX.TRANSLATE('Welcome to Snowflake Data Cloud', 'en', 'fr') AS french,
    SNOWFLAKE.CORTEX.TRANSLATE('Welcome to Snowflake Data Cloud', 'en', 'de') AS german;

-- SECTION 7: EXTRACT_ANSWER for Question-Answering
SELECT SNOWFLAKE.CORTEX.EXTRACT_ANSWER(
    'Snowflake was founded in 2012 by Benoit Dageville, Thierry Cruanes, and 
    Marcin Zukowski. The company is headquartered in Bozeman, Montana. 
    Snowflake went public on September 16, 2020, with the largest software 
    IPO in history at that time, raising $3.4 billion.',
    'When was Snowflake founded and by whom?'
) AS extracted_answer;

-- SECTION 8: Practical Application - Automated Ticket Classification
CREATE OR REPLACE TEMPORARY TABLE support_tickets (
    ticket_id INT,
    subject VARCHAR(200),
    description TEXT
);

INSERT INTO support_tickets VALUES
    (1001, 'Query running slow', 'My query has been running for 2 hours on a LARGE warehouse. The query profile shows remote spillage.'),
    (1002, 'Cannot login', 'Getting authentication error when trying to connect. MFA is enabled on my account.'),
    (1003, 'Data not showing', 'The dashboard is not showing data from yesterday. Snowpipe might have failed.'),
    (1004, 'Need more credits', 'We are running low on credits and need to increase our capacity.');

SELECT 
    ticket_id,
    subject,
    SNOWFLAKE.CORTEX.COMPLETE(
        'mistral-large',
        [
            {'role': 'system', 'content': 'Classify the support ticket into exactly one category: PERFORMANCE, AUTHENTICATION, DATA_PIPELINE, BILLING. Return only the category name.'},
            {'role': 'user', 'content': description}
        ],
        {'temperature': 0, 'max_tokens': 20}
    ) AS category,
    SNOWFLAKE.CORTEX.SENTIMENT(description) AS urgency_score
FROM support_tickets;

-- SECTION 9: Model Comparison
-- Compare responses from different models
SELECT 
    'mistral-large' AS model,
    SNOWFLAKE.CORTEX.COMPLETE('mistral-large', 'What is Time Travel in Snowflake? (1 sentence)') AS response
UNION ALL
SELECT 
    'llama3-70b' AS model,
    SNOWFLAKE.CORTEX.COMPLETE('llama3-70b', 'What is Time Travel in Snowflake? (1 sentence)') AS response
UNION ALL
SELECT 
    'mistral-7b' AS model,
    SNOWFLAKE.CORTEX.COMPLETE('mistral-7b', 'What is Time Travel in Snowflake? (1 sentence)') AS response;

-- SECTION 10: Cleanup (uncomment to run)
-- DROP TABLE customer_reviews;
-- DROP TABLE support_tickets;
-- DROP SCHEMA llm_demo;
-- DROP DATABASE cortex_lab;
""",

        "Cortex Search Setup": """
-- ============================================
-- SQL Lab: Cortex Search Service for RAG
-- Based on Snowflake Documentation Best Practices
-- ============================================

-- SECTION 1: Environment Setup
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;
CREATE DATABASE IF NOT EXISTS cortex_search_lab;
USE DATABASE cortex_search_lab;
CREATE SCHEMA IF NOT EXISTS rag_demo;
USE SCHEMA rag_demo;

-- SECTION 2: Create Knowledge Base Table
CREATE OR REPLACE TABLE knowledge_base (
    doc_id INT AUTOINCREMENT,
    title VARCHAR(500),
    content TEXT,
    category VARCHAR(100),
    source_url VARCHAR(1000),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- SECTION 3: Insert Sample Documentation
INSERT INTO knowledge_base (title, content, category, source_url) VALUES
    ('Virtual Warehouses Overview', 
     'Virtual warehouses are clusters of compute resources in Snowflake. They execute queries and DML operations. Warehouses can be scaled up (larger size) or scaled out (multi-cluster). Warehouses automatically suspend after a period of inactivity and resume when queries are submitted. Credit consumption is based on warehouse size and runtime.',
     'Compute', 'https://docs.snowflake.com/en/user-guide/warehouses'),
    
    ('Time Travel and Fail-safe',
     'Time Travel enables accessing historical data at any point within a defined period. Standard Edition supports 1 day, Enterprise up to 90 days. Use AT or BEFORE clause to query historical data. UNDROP recovers dropped objects. Fail-safe provides additional 7 days of recovery through Snowflake support.',
     'Data Protection', 'https://docs.snowflake.com/en/user-guide/data-time-travel'),
    
    ('Data Sharing Concepts',
     'Secure Data Sharing allows sharing live data without copying. Providers create shares and grant access to consumers. Consumers create databases from shares. Data is always current - no sync needed. Reader accounts enable sharing with non-Snowflake customers.',
     'Collaboration', 'https://docs.snowflake.com/en/user-guide/data-sharing'),
    
    ('Micro-partitions and Clustering',
     'Snowflake stores data in micro-partitions of 50-500MB compressed. Each partition stores metadata about min/max values. Clustering keys define data organization. Automatic clustering maintains partitions. Query pruning skips irrelevant partitions based on predicates.',
     'Storage', 'https://docs.snowflake.com/en/user-guide/tables-clustering-micropartitions'),
    
    ('Streams for Change Data Capture',
     'Streams track DML changes (INSERT, UPDATE, DELETE) on tables. Standard streams track all changes with METADATA$ACTION and METADATA$ISUPDATE columns. Append-only streams track inserts only. Streams are consumed when data is read in a DML transaction.',
     'Data Engineering', 'https://docs.snowflake.com/en/user-guide/streams'),
    
    ('Cortex LLM Functions',
     'Snowflake Cortex provides serverless LLM functions. COMPLETE generates text using models like mistral-large and llama3-70b. SUMMARIZE condenses long text. SENTIMENT returns scores from -1 to 1. TRANSLATE converts between languages. EXTRACT_ANSWER finds answers in context.',
     'AI/ML', 'https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions');

-- SECTION 4: Create Cortex Search Service
CREATE OR REPLACE CORTEX SEARCH SERVICE knowledge_search
    ON knowledge_base
    ATTRIBUTES category, title
    WAREHOUSE = COMPUTE_WH
    TARGET_LAG = '1 hour';

-- Wait for initial indexing (check status)
SELECT SYSTEM$CORTEX_SEARCH_SERVICE_STATUS('knowledge_search');

-- SECTION 5: Query the Search Service
-- Basic semantic search
SELECT SEARCH(
    'knowledge_search', 
    'How do I protect my data from accidental deletion?',
    {'limit': 3}
) AS search_results;

-- SECTION 6: RAG Implementation - Combine Search with LLM
-- Step 1: Search for relevant context
WITH search_context AS (
    SELECT SEARCH(
        'knowledge_search',
        'What are best practices for warehouse sizing?',
        {'limit': 3}
    ) AS results
),
-- Step 2: Extract content from search results
relevant_docs AS (
    SELECT 
        f.value:content::STRING AS doc_content,
        f.value:title::STRING AS doc_title
    FROM search_context, 
    LATERAL FLATTEN(input => results:results) f
)
-- Step 3: Generate grounded response using LLM
SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'mistral-large',
    [
        {'role': 'system', 'content': 'You are a Snowflake expert. Answer based ONLY on the provided context. If the context does not contain the answer, say so.'},
        {'role': 'user', 'content': 'Context:\n' || LISTAGG(doc_content, '\n\n') || '\n\nQuestion: What are best practices for warehouse sizing?'}
    ],
    {'temperature': 0.2, 'max_tokens': 500}
) AS rag_response
FROM relevant_docs;

-- SECTION 7: Filter by Category
SELECT SEARCH(
    'knowledge_search',
    'How does Snowflake handle compute?',
    {
        'limit': 5,
        'filter': {'category': 'Compute'}
    }
) AS filtered_results;

-- SECTION 8: Monitor Search Service
-- Check service status
SHOW CORTEX SEARCH SERVICES;

-- View refresh history
SELECT * FROM TABLE(INFORMATION_SCHEMA.CORTEX_SEARCH_SERVICE_REFRESH_HISTORY())
WHERE SERVICE_NAME = 'KNOWLEDGE_SEARCH'
ORDER BY REFRESH_START_TIME DESC
LIMIT 10;

-- SECTION 9: Cleanup (uncomment to run)
-- DROP CORTEX SEARCH SERVICE knowledge_search;
-- DROP TABLE knowledge_base;
-- DROP SCHEMA rag_demo;
-- DROP DATABASE cortex_search_lab;
""",

        "Vector Embeddings with EMBED_TEXT": """
-- ============================================
-- SQL Lab: Vector Embeddings and Semantic Search
-- Based on Snowflake Documentation Best Practices
-- ============================================

-- SECTION 1: Environment Setup
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;
CREATE DATABASE IF NOT EXISTS vector_lab;
USE DATABASE vector_lab;
CREATE SCHEMA IF NOT EXISTS embeddings;
USE SCHEMA embeddings;

-- SECTION 2: Create Table with VECTOR Column
CREATE OR REPLACE TABLE documents (
    doc_id INT AUTOINCREMENT,
    title VARCHAR(500),
    content TEXT,
    category VARCHAR(100),
    embedding VECTOR(FLOAT, 768),  -- 768 dimensions for e5-base-v2
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- SECTION 3: Insert Documents with Embeddings
INSERT INTO documents (title, content, category, embedding)
SELECT 
    'Snowflake Architecture',
    'Snowflake uses a multi-cluster shared data architecture with three layers: storage, compute, and cloud services. This separation enables independent scaling.',
    'Architecture',
    SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', 
        'Snowflake uses a multi-cluster shared data architecture with three layers: storage, compute, and cloud services. This separation enables independent scaling.');

INSERT INTO documents (title, content, category, embedding)
SELECT 
    'Virtual Warehouses',
    'Virtual warehouses are compute clusters that execute queries. They can scale up by size or scale out with multi-cluster. Auto-suspend reduces costs during idle periods.',
    'Compute',
    SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2',
        'Virtual warehouses are compute clusters that execute queries. They can scale up by size or scale out with multi-cluster. Auto-suspend reduces costs during idle periods.');

INSERT INTO documents (title, content, category, embedding)
SELECT 
    'Data Loading with Snowpipe',
    'Snowpipe provides serverless, continuous data loading. It automatically ingests files when they arrive in cloud storage using event notifications.',
    'Data Engineering',
    SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2',
        'Snowpipe provides serverless, continuous data loading. It automatically ingests files when they arrive in cloud storage using event notifications.');

INSERT INTO documents (title, content, category, embedding)
SELECT 
    'Secure Data Sharing',
    'Secure Data Sharing enables sharing live data without copying. Providers grant access to consumers who can query data in real-time. No data movement required.',
    'Collaboration',
    SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2',
        'Secure Data Sharing enables sharing live data without copying. Providers grant access to consumers who can query data in real-time. No data movement required.');

INSERT INTO documents (title, content, category, embedding)
SELECT 
    'Cortex AI Functions',
    'Snowflake Cortex provides LLM functions like COMPLETE, SUMMARIZE, SENTIMENT, and TRANSLATE. These run serverlessly on Snowflake compute without external API calls.',
    'AI/ML',
    SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2',
        'Snowflake Cortex provides LLM functions like COMPLETE, SUMMARIZE, SENTIMENT, and TRANSLATE. These run serverlessly on Snowflake compute without external API calls.');

-- SECTION 4: Semantic Search with Cosine Similarity
-- Find documents similar to a query
SET query_text = 'How can I load data continuously into Snowflake?';

WITH query_embedding AS (
    SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', $query_text) AS query_vec
)
SELECT 
    d.title,
    d.content,
    d.category,
    VECTOR_COSINE_SIMILARITY(d.embedding, q.query_vec) AS similarity_score
FROM documents d
CROSS JOIN query_embedding q
ORDER BY similarity_score DESC
LIMIT 3;

-- SECTION 5: Different Similarity Metrics
SET search_query = 'scaling compute resources';

WITH query_emb AS (
    SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', $search_query) AS vec
)
SELECT 
    d.title,
    VECTOR_COSINE_SIMILARITY(d.embedding, q.vec) AS cosine_sim,
    VECTOR_L2_DISTANCE(d.embedding, q.vec) AS l2_distance,
    VECTOR_INNER_PRODUCT(d.embedding, q.vec) AS inner_product
FROM documents d
CROSS JOIN query_emb q
ORDER BY cosine_sim DESC;

-- SECTION 6: RAG Pattern - Retrieve and Generate
SET user_question = 'What is the best way to share data with external partners?';

WITH 
-- Step 1: Embed the question
question_embedding AS (
    SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', $user_question) AS q_vec
),
-- Step 2: Find most relevant documents
relevant_docs AS (
    SELECT 
        d.title,
        d.content,
        VECTOR_COSINE_SIMILARITY(d.embedding, q.q_vec) AS score
    FROM documents d
    CROSS JOIN question_embedding q
    ORDER BY score DESC
    LIMIT 2
),
-- Step 3: Build context from top documents
context AS (
    SELECT LISTAGG(content, '\n\n') AS combined_context
    FROM relevant_docs
)
-- Step 4: Generate answer using LLM
SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'mistral-large',
    [
        {'role': 'system', 'content': 'You are a helpful Snowflake assistant. Answer based only on the provided context.'},
        {'role': 'user', 'content': 'Context: ' || combined_context || '\n\nQuestion: ' || $user_question}
    ],
    {'temperature': 0.2}
) AS rag_answer
FROM context;

-- SECTION 7: Batch Embedding for Large Datasets
-- Efficient pattern for embedding many documents
CREATE OR REPLACE TABLE documents_to_embed (
    doc_id INT,
    text_content TEXT
);

INSERT INTO documents_to_embed VALUES
    (1, 'First document to embed'),
    (2, 'Second document to embed'),
    (3, 'Third document to embed');

-- Batch embed using SELECT INTO
CREATE OR REPLACE TABLE embedded_documents AS
SELECT 
    doc_id,
    text_content,
    SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', text_content) AS embedding
FROM documents_to_embed;

SELECT * FROM embedded_documents;

-- SECTION 8: Cleanup (uncomment to run)
-- DROP TABLE documents;
-- DROP TABLE documents_to_embed;
-- DROP TABLE embedded_documents;
-- DROP SCHEMA embeddings;
-- DROP DATABASE vector_lab;
""",

        "Streams and Tasks": """
-- ============================================
-- SQL Lab: Streams and Tasks for CDC Pipelines
-- Based on Snowflake Documentation Best Practices
-- ============================================

-- SECTION 1: Environment Setup
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;
CREATE DATABASE IF NOT EXISTS pipeline_lab;
USE DATABASE pipeline_lab;
CREATE SCHEMA IF NOT EXISTS cdc_demo;
USE SCHEMA cdc_demo;

-- SECTION 2: Create Source and Target Tables
CREATE OR REPLACE TABLE raw_orders (
    order_id INT,
    customer_id INT,
    product_id INT,
    quantity INT,
    unit_price DECIMAL(10,2),
    order_status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

CREATE OR REPLACE TABLE processed_orders (
    order_id INT,
    customer_id INT,
    product_id INT,
    quantity INT,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    order_status VARCHAR(50),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    change_type VARCHAR(20)
);

CREATE OR REPLACE TABLE order_audit_log (
    log_id INT AUTOINCREMENT,
    order_id INT,
    action VARCHAR(20),
    old_status VARCHAR(50),
    new_status VARCHAR(50),
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- SECTION 3: Create Stream on Source Table
-- Standard stream captures INSERT, UPDATE, DELETE
CREATE OR REPLACE STREAM orders_stream ON TABLE raw_orders;

-- SECTION 4: Insert Initial Data
INSERT INTO raw_orders (order_id, customer_id, product_id, quantity, unit_price)
VALUES 
    (1001, 101, 501, 2, 29.99),
    (1002, 102, 502, 1, 149.99),
    (1003, 103, 501, 5, 29.99);

-- SECTION 5: View Stream Contents
-- Stream shows CDC records with metadata
SELECT 
    order_id,
    customer_id,
    order_status,
    METADATA$ACTION AS change_action,
    METADATA$ISUPDATE AS is_update,
    METADATA$ROW_ID AS row_id
FROM orders_stream;

-- SECTION 6: Create Stored Procedure for Processing
CREATE OR REPLACE PROCEDURE process_order_changes()
RETURNS STRING
LANGUAGE SQL
AS
$$
DECLARE
    rows_processed INT DEFAULT 0;
BEGIN
    -- Process inserts and updates
    INSERT INTO processed_orders (
        order_id, customer_id, product_id, quantity, 
        unit_price, total_amount, order_status, change_type
    )
    SELECT 
        order_id,
        customer_id,
        product_id,
        quantity,
        unit_price,
        quantity * unit_price AS total_amount,
        order_status,
        CASE 
            WHEN METADATA$ISUPDATE THEN 'UPDATE'
            ELSE METADATA$ACTION
        END AS change_type
    FROM orders_stream
    WHERE METADATA$ACTION = 'INSERT';
    
    rows_processed := SQLROWCOUNT;
    
    -- Log status changes
    INSERT INTO order_audit_log (order_id, action, new_status)
    SELECT order_id, 'STATUS_CHANGE', order_status
    FROM orders_stream
    WHERE METADATA$ISUPDATE = TRUE;
    
    RETURN 'Processed ' || rows_processed || ' records';
END;
$$;

-- SECTION 7: Create Task to Run Automatically
CREATE OR REPLACE TASK process_orders_task
    WAREHOUSE = COMPUTE_WH
    SCHEDULE = '1 MINUTE'
    WHEN SYSTEM$STREAM_HAS_DATA('orders_stream')
AS
    CALL process_order_changes();

-- Enable the task
ALTER TASK process_orders_task RESUME;

-- SECTION 8: Simulate More Changes
-- Insert new orders
INSERT INTO raw_orders (order_id, customer_id, product_id, quantity, unit_price)
VALUES 
    (1004, 104, 503, 3, 79.99),
    (1005, 101, 504, 1, 299.99);

-- Update existing order
UPDATE raw_orders 
SET order_status = 'shipped', updated_at = CURRENT_TIMESTAMP()
WHERE order_id = 1001;

-- SECTION 9: Monitor Task Execution
-- View task history
SELECT 
    NAME,
    STATE,
    SCHEDULED_TIME,
    COMPLETED_TIME,
    ERROR_MESSAGE
FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
    SCHEDULED_TIME_RANGE_START => DATEADD('hour', -1, CURRENT_TIMESTAMP()),
    TASK_NAME => 'PROCESS_ORDERS_TASK'
))
ORDER BY SCHEDULED_TIME DESC
LIMIT 10;

-- Check processed results
SELECT * FROM processed_orders ORDER BY processed_at DESC;
SELECT * FROM order_audit_log ORDER BY logged_at DESC;

-- SECTION 10: Task DAG Example
-- Create child tasks that depend on parent
CREATE OR REPLACE TASK parent_task
    WAREHOUSE = COMPUTE_WH
    SCHEDULE = '5 MINUTE'
AS
    SELECT 'Parent task executed' AS status;

CREATE OR REPLACE TASK child_task_1
    WAREHOUSE = COMPUTE_WH
    AFTER parent_task
AS
    SELECT 'Child 1 executed' AS status;

CREATE OR REPLACE TASK child_task_2
    WAREHOUSE = COMPUTE_WH
    AFTER parent_task
AS
    SELECT 'Child 2 executed' AS status;

CREATE OR REPLACE TASK grandchild_task
    WAREHOUSE = COMPUTE_WH
    AFTER child_task_1, child_task_2
AS
    SELECT 'Grandchild executed after both children' AS status;

-- Enable DAG (start from leaves, then parent)
ALTER TASK grandchild_task RESUME;
ALTER TASK child_task_2 RESUME;
ALTER TASK child_task_1 RESUME;
ALTER TASK parent_task RESUME;

-- SECTION 11: Cleanup (uncomment to run)
-- ALTER TASK parent_task SUSPEND;
-- ALTER TASK child_task_1 SUSPEND;
-- ALTER TASK child_task_2 SUSPEND;
-- ALTER TASK grandchild_task SUSPEND;
-- ALTER TASK process_orders_task SUSPEND;
-- DROP TASK grandchild_task;
-- DROP TASK child_task_2;
-- DROP TASK child_task_1;
-- DROP TASK parent_task;
-- DROP TASK process_orders_task;
-- DROP STREAM orders_stream;
-- DROP PROCEDURE process_order_changes();
-- DROP TABLE order_audit_log;
-- DROP TABLE processed_orders;
-- DROP TABLE raw_orders;
-- DROP SCHEMA cdc_demo;
-- DROP DATABASE pipeline_lab;
""",

        "Dynamic Tables": """
-- ============================================
-- SQL Lab: Dynamic Tables for Declarative Pipelines
-- Based on Snowflake Documentation Best Practices
-- ============================================

-- SECTION 1: Environment Setup
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;
CREATE DATABASE IF NOT EXISTS dynamic_tables_lab;
USE DATABASE dynamic_tables_lab;
CREATE SCHEMA IF NOT EXISTS pipeline;
USE SCHEMA pipeline;

-- SECTION 2: Create Source Tables (Bronze Layer)
CREATE OR REPLACE TABLE raw_transactions (
    txn_id VARCHAR(50),
    customer_id INT,
    product_id INT,
    quantity INT,
    unit_price DECIMAL(10,2),
    txn_timestamp TIMESTAMP,
    store_id INT,
    payment_method VARCHAR(50)
);

CREATE OR REPLACE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(200),
    category VARCHAR(100),
    brand VARCHAR(100)
);

CREATE OR REPLACE TABLE stores (
    store_id INT PRIMARY KEY,
    store_name VARCHAR(200),
    region VARCHAR(100),
    country VARCHAR(100)
);

-- SECTION 3: Insert Sample Data
INSERT INTO products VALUES
    (1, 'Laptop Pro 15', 'Electronics', 'TechBrand'),
    (2, 'Wireless Mouse', 'Electronics', 'TechBrand'),
    (3, 'Office Chair', 'Furniture', 'ComfortCo'),
    (4, 'Standing Desk', 'Furniture', 'ComfortCo'),
    (5, 'Monitor 27inch', 'Electronics', 'ViewMax');

INSERT INTO stores VALUES
    (101, 'Downtown Store', 'Northeast', 'USA'),
    (102, 'Mall Location', 'Southeast', 'USA'),
    (103, 'Tech Hub', 'West', 'USA'),
    (104, 'Online Store', 'Digital', 'USA');

INSERT INTO raw_transactions VALUES
    ('TXN001', 1001, 1, 1, 1299.99, '2024-01-15 10:30:00', 101, 'Credit Card'),
    ('TXN002', 1002, 2, 2, 49.99, '2024-01-15 11:45:00', 102, 'Debit Card'),
    ('TXN003', 1001, 5, 2, 399.99, '2024-01-15 14:20:00', 101, 'Credit Card'),
    ('TXN004', 1003, 3, 1, 299.99, '2024-01-16 09:15:00', 103, 'PayPal'),
    ('TXN005', 1004, 4, 1, 599.99, '2024-01-16 13:30:00', 104, 'Credit Card'),
    ('TXN006', 1002, 1, 1, 1299.99, '2024-01-17 10:00:00', 102, 'Credit Card');

-- SECTION 4: Silver Layer - Enriched Transactions
CREATE OR REPLACE DYNAMIC TABLE enriched_transactions
    TARGET_LAG = '1 minute'
    WAREHOUSE = COMPUTE_WH
AS
SELECT 
    t.txn_id,
    t.customer_id,
    t.txn_timestamp,
    t.quantity,
    t.unit_price,
    t.quantity * t.unit_price AS total_amount,
    t.payment_method,
    p.product_name,
    p.category,
    p.brand,
    s.store_name,
    s.region,
    s.country,
    DATE(t.txn_timestamp) AS txn_date
FROM raw_transactions t
JOIN products p ON t.product_id = p.product_id
JOIN stores s ON t.store_id = s.store_id;

-- SECTION 5: Gold Layer - Daily Sales Summary
CREATE OR REPLACE DYNAMIC TABLE daily_sales_summary
    TARGET_LAG = '5 minutes'
    WAREHOUSE = COMPUTE_WH
AS
SELECT 
    txn_date,
    region,
    category,
    COUNT(DISTINCT txn_id) AS num_transactions,
    COUNT(DISTINCT customer_id) AS unique_customers,
    SUM(quantity) AS total_units_sold,
    SUM(total_amount) AS total_revenue,
    AVG(total_amount) AS avg_transaction_value
FROM enriched_transactions
GROUP BY txn_date, region, category;

-- SECTION 6: Gold Layer - Customer Metrics
CREATE OR REPLACE DYNAMIC TABLE customer_metrics
    TARGET_LAG = '10 minutes'
    WAREHOUSE = COMPUTE_WH
AS
SELECT 
    customer_id,
    COUNT(DISTINCT txn_id) AS lifetime_transactions,
    SUM(total_amount) AS lifetime_value,
    MIN(txn_date) AS first_purchase_date,
    MAX(txn_date) AS last_purchase_date,
    DATEDIFF('day', MIN(txn_date), MAX(txn_date)) AS customer_tenure_days,
    COUNT(DISTINCT category) AS categories_purchased
FROM enriched_transactions
GROUP BY customer_id;

-- SECTION 7: Query the Dynamic Tables
SELECT * FROM enriched_transactions ORDER BY txn_timestamp DESC;

SELECT * FROM daily_sales_summary ORDER BY txn_date DESC, total_revenue DESC;

SELECT * FROM customer_metrics ORDER BY lifetime_value DESC;

-- SECTION 8: Add More Data and Observe Refresh
INSERT INTO raw_transactions VALUES
    ('TXN007', 1005, 1, 2, 1299.99, CURRENT_TIMESTAMP(), 101, 'Credit Card'),
    ('TXN008', 1001, 2, 3, 49.99, CURRENT_TIMESTAMP(), 103, 'Debit Card');

-- Check refresh status
SELECT 
    NAME,
    SCHEMA_NAME,
    TARGET_LAG,
    REFRESH_MODE,
    SCHEDULING_STATE
FROM INFORMATION_SCHEMA.DYNAMIC_TABLES
WHERE TABLE_SCHEMA = 'PIPELINE';

-- SECTION 9: Monitor Refresh History
SELECT 
    NAME,
    REFRESH_START_TIME,
    REFRESH_END_TIME,
    REFRESH_ACTION,
    STATE,
    DATA_TIMESTAMP
FROM TABLE(INFORMATION_SCHEMA.DYNAMIC_TABLE_REFRESH_HISTORY())
WHERE NAME IN ('ENRICHED_TRANSACTIONS', 'DAILY_SALES_SUMMARY', 'CUSTOMER_METRICS')
ORDER BY REFRESH_START_TIME DESC
LIMIT 20;

-- SECTION 10: Pipeline Lineage
-- View how dynamic tables depend on each other
SELECT 
    NAME,
    TEXT AS definition_query
FROM INFORMATION_SCHEMA.DYNAMIC_TABLES
WHERE TABLE_SCHEMA = 'PIPELINE';

-- SECTION 11: Cleanup (uncomment to run)
-- DROP DYNAMIC TABLE customer_metrics;
-- DROP DYNAMIC TABLE daily_sales_summary;
-- DROP DYNAMIC TABLE enriched_transactions;
-- DROP TABLE raw_transactions;
-- DROP TABLE products;
-- DROP TABLE stores;
-- DROP SCHEMA pipeline;
-- DROP DATABASE dynamic_tables_lab;
""",

        "Cortex Guard Implementation": """
-- ============================================
-- SQL Lab: Cortex Guard for AI Safety
-- Based on Snowflake Documentation Best Practices
-- ============================================

-- SECTION 1: Environment Setup
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;
CREATE DATABASE IF NOT EXISTS cortex_guard_lab;
USE DATABASE cortex_guard_lab;
CREATE SCHEMA IF NOT EXISTS ai_safety;
USE SCHEMA ai_safety;

-- SECTION 2: Understanding Cortex Guard
-- Cortex Guard filters harmful content in AI applications
-- Categories: hate, violence, sexual, self-harm, dangerous

-- Basic Cortex Guard check
SELECT SNOWFLAKE.CORTEX.GUARD(
    'mistral-large',
    'What is the weather like today?'
) AS safety_check;

-- SECTION 3: Test with Various Inputs
CREATE OR REPLACE TEMPORARY TABLE test_inputs (
    input_id INT,
    user_input TEXT,
    expected_safe BOOLEAN
);

INSERT INTO test_inputs VALUES
    (1, 'How do I optimize my Snowflake queries?', TRUE),
    (2, 'What are best practices for data security?', TRUE),
    (3, 'Explain virtual warehouses in simple terms', TRUE),
    (4, 'Help me write a marketing email for our product', TRUE);

-- Check each input with Cortex Guard
SELECT 
    input_id,
    user_input,
    expected_safe,
    SNOWFLAKE.CORTEX.GUARD('mistral-large', user_input) AS guard_result,
    PARSE_JSON(SNOWFLAKE.CORTEX.GUARD('mistral-large', user_input)):safe::BOOLEAN AS is_safe
FROM test_inputs;

-- SECTION 4: Implementing Input/Output Filtering
-- Create a safe chatbot function
CREATE OR REPLACE FUNCTION safe_chatbot(user_message TEXT)
RETURNS TABLE (response TEXT, filtered BOOLEAN)
LANGUAGE SQL
AS
$$
    WITH 
    -- Step 1: Check if input is safe
    input_check AS (
        SELECT 
            user_message AS msg,
            PARSE_JSON(SNOWFLAKE.CORTEX.GUARD('mistral-large', user_message)):safe::BOOLEAN AS input_safe
    ),
    -- Step 2: Generate response only if input is safe
    llm_response AS (
        SELECT 
            CASE 
                WHEN input_safe THEN 
                    SNOWFLAKE.CORTEX.COMPLETE(
                        'mistral-large',
                        [
                            {'role': 'system', 'content': 'You are a helpful Snowflake assistant.'},
                            {'role': 'user', 'content': msg}
                        ],
                        {'temperature': 0.3}
                    )
                ELSE 'I cannot process this request.'
            END AS raw_response,
            input_safe
        FROM input_check
    ),
    -- Step 3: Check if output is safe
    output_check AS (
        SELECT 
            raw_response,
            input_safe,
            CASE 
                WHEN input_safe THEN
                    PARSE_JSON(SNOWFLAKE.CORTEX.GUARD('mistral-large', raw_response)):safe::BOOLEAN
                ELSE FALSE
            END AS output_safe
        FROM llm_response
    )
    -- Step 4: Return safe response or filtered message
    SELECT 
        CASE 
            WHEN NOT input_safe THEN 'Your input was filtered for safety.'
            WHEN NOT output_safe THEN 'The response was filtered for safety.'
            ELSE raw_response
        END AS response,
        NOT (input_safe AND output_safe) AS filtered
    FROM output_check
$$;

-- Test the safe chatbot
SELECT * FROM TABLE(safe_chatbot('Explain data sharing in Snowflake'));

-- SECTION 5: Logging Filtered Content for Review
CREATE OR REPLACE TABLE content_filter_log (
    log_id INT AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    content_type VARCHAR(10),  -- 'INPUT' or 'OUTPUT'
    content_hash VARCHAR(64),  -- Hash of content (don't store actual content)
    guard_result VARIANT,
    is_safe BOOLEAN
);

-- Create logging procedure
CREATE OR REPLACE PROCEDURE log_and_filter(content TEXT, content_type TEXT)
RETURNS VARIANT
LANGUAGE SQL
AS
$$
DECLARE
    guard_result VARIANT;
    is_safe BOOLEAN;
BEGIN
    -- Get guard result
    guard_result := PARSE_JSON(SNOWFLAKE.CORTEX.GUARD('mistral-large', :content));
    is_safe := guard_result:safe::BOOLEAN;
    
    -- Log the check (store hash, not actual content)
    INSERT INTO content_filter_log (content_type, content_hash, guard_result, is_safe)
    VALUES (:content_type, SHA2(:content), :guard_result, :is_safe);
    
    RETURN guard_result;
END;
$$;

-- SECTION 6: View Filter Statistics
-- After running many checks, analyze patterns
SELECT 
    DATE(timestamp) AS check_date,
    content_type,
    COUNT(*) AS total_checks,
    SUM(CASE WHEN is_safe THEN 1 ELSE 0 END) AS safe_count,
    SUM(CASE WHEN NOT is_safe THEN 1 ELSE 0 END) AS filtered_count,
    ROUND(SUM(CASE WHEN NOT is_safe THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS filter_rate_pct
FROM content_filter_log
GROUP BY DATE(timestamp), content_type
ORDER BY check_date DESC;

-- SECTION 7: Integration with Chat Application
-- Complete flow: Input Guard -> LLM -> Output Guard
CREATE OR REPLACE FUNCTION guarded_complete(
    system_prompt TEXT,
    user_message TEXT
)
RETURNS OBJECT
LANGUAGE SQL
AS
$$
    SELECT OBJECT_CONSTRUCT(
        'input_safe', PARSE_JSON(SNOWFLAKE.CORTEX.GUARD('mistral-large', user_message)):safe::BOOLEAN,
        'response', 
            CASE 
                WHEN PARSE_JSON(SNOWFLAKE.CORTEX.GUARD('mistral-large', user_message)):safe::BOOLEAN
                THEN SNOWFLAKE.CORTEX.COMPLETE(
                    'mistral-large',
                    [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_message}
                    ],
                    {'temperature': 0.3}
                )
                ELSE 'Message filtered for safety'
            END,
        'timestamp', CURRENT_TIMESTAMP()
    )
$$;

-- Test the guarded function
SELECT guarded_complete(
    'You are a Snowflake expert assistant.',
    'What are the benefits of using Snowflake Cortex?'
) AS result;

-- SECTION 8: Cleanup (uncomment to run)
-- DROP FUNCTION IF EXISTS safe_chatbot(TEXT);
-- DROP FUNCTION IF EXISTS guarded_complete(TEXT, TEXT);
-- DROP PROCEDURE IF EXISTS log_and_filter(TEXT, TEXT);
-- DROP TABLE IF EXISTS content_filter_log;
-- DROP TABLE IF EXISTS test_inputs;
-- DROP SCHEMA ai_safety;
-- DROP DATABASE cortex_guard_lab;
""",

        "Document AI Setup and PREDICT Queries": """
-- ============================================
-- SQL Lab: Document AI for Intelligent Extraction
-- Based on Snowflake Documentation Best Practices
-- ============================================

-- SECTION 1: Environment Setup
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;
CREATE DATABASE IF NOT EXISTS document_ai_lab;
USE DATABASE document_ai_lab;
CREATE SCHEMA IF NOT EXISTS extraction;
USE SCHEMA extraction;

-- SECTION 2: Create Stage for Documents
CREATE OR REPLACE STAGE document_stage
    DIRECTORY = (ENABLE = TRUE);

-- SECTION 3: Using PARSE_DOCUMENT for General Extraction
-- Note: Upload a PDF to the stage first, or use the examples below

-- Parse document with layout preservation
-- SELECT SNOWFLAKE.CORTEX.PARSE_DOCUMENT(
--     '@document_stage/sample_invoice.pdf',
--     {'mode': 'LAYOUT'}
-- ) AS parsed_content;

-- Parse with OCR for image-based documents
-- SELECT SNOWFLAKE.CORTEX.PARSE_DOCUMENT(
--     '@document_stage/scanned_form.pdf',
--     {'mode': 'OCR'}
-- ) AS ocr_content;

-- SECTION 4: Create Table to Store Extracted Documents
CREATE OR REPLACE TABLE extracted_documents (
    doc_id INT AUTOINCREMENT,
    file_name VARCHAR(500),
    file_path VARCHAR(1000),
    extracted_text TEXT,
    extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    metadata VARIANT
);

-- SECTION 5: Document AI Build Workflow (Conceptual)
-- Step 1: Create a Document AI build
-- CREATE DOCUMENT AI BUILD invoice_extractor
--     WAREHOUSE = COMPUTE_WH;

-- Step 2: Upload training documents to the build
-- (Done through Snowsight UI or API)

-- Step 3: Define extraction questions
-- Questions like:
-- - "What is the invoice number?"
-- - "What is the total amount?"
-- - "What is the vendor name?"
-- - "What is the invoice date?"

-- Step 4: Train the model
-- ALTER DOCUMENT AI BUILD invoice_extractor TRAIN;

-- Step 5: Deploy for predictions
-- ALTER DOCUMENT AI BUILD invoice_extractor DEPLOY;

-- SECTION 6: Using !PREDICT for Extraction (After Training)
-- Once Document AI is trained and deployed:
-- SELECT 
--     file_path,
--     invoice_extractor!PREDICT(GET_PRESIGNED_URL(@document_stage, file_path)) AS extracted_data
-- FROM directory(@document_stage)
-- WHERE file_path LIKE '%.pdf';

-- SECTION 7: GET_PRESIGNED_URL for Document Access
-- Generate presigned URLs for documents in stage
SELECT 
    GET_PRESIGNED_URL(@document_stage, 'sample.pdf', 3600) AS presigned_url;

-- SECTION 8: Batch Processing Pipeline Pattern
-- Create pipeline for processing uploaded documents
CREATE OR REPLACE STREAM new_documents_stream 
    ON STAGE document_stage;

CREATE OR REPLACE TABLE processed_documents (
    doc_id INT AUTOINCREMENT,
    file_name VARCHAR(500),
    upload_time TIMESTAMP,
    extracted_fields VARIANT,
    processing_status VARCHAR(50)
);

-- Task to process new documents
CREATE OR REPLACE TASK process_new_documents
    WAREHOUSE = COMPUTE_WH
    SCHEDULE = '5 MINUTE'
    -- WHEN SYSTEM$STREAM_HAS_DATA('new_documents_stream')
AS
    INSERT INTO processed_documents (file_name, upload_time, processing_status)
    SELECT 
        METADATA$FILENAME,
        CURRENT_TIMESTAMP(),
        'QUEUED'
    FROM new_documents_stream
    WHERE METADATA$ACTION = 'INSERT'
    AND METADATA$FILENAME LIKE '%.pdf';

-- SECTION 9: LLM-Based Document Processing Alternative
-- Use COMPLETE for flexible extraction without training
CREATE OR REPLACE FUNCTION extract_invoice_fields(document_text TEXT)
RETURNS VARIANT
LANGUAGE SQL
AS
$$
    SELECT PARSE_JSON(SNOWFLAKE.CORTEX.COMPLETE(
        'mistral-large',
        [
            {
                'role': 'system', 
                'content': 'Extract the following fields from the invoice text and return as JSON: invoice_number, date, vendor_name, total_amount, line_items (array with description, quantity, unit_price). Return ONLY valid JSON.'
            },
            {'role': 'user', 'content': document_text}
        ],
        {'temperature': 0}
    ))
$$;

-- Test with sample invoice text
SELECT extract_invoice_fields(
    'INVOICE #INV-2024-001\n' ||
    'Date: January 15, 2024\n' ||
    'Vendor: Acme Corp\n\n' ||
    'Items:\n' ||
    '1. Widget A - Qty: 10 @ $25.00 = $250.00\n' ||
    '2. Widget B - Qty: 5 @ $45.00 = $225.00\n\n' ||
    'Total: $475.00'
) AS extracted_fields;

-- SECTION 10: Error Handling for Document Processing
CREATE OR REPLACE PROCEDURE process_document_safe(file_path TEXT)
RETURNS VARIANT
LANGUAGE SQL
AS
$$
DECLARE
    result VARIANT;
    error_msg TEXT;
BEGIN
    BEGIN
        -- Attempt to parse document
        -- result := SNOWFLAKE.CORTEX.PARSE_DOCUMENT(file_path, {'mode': 'LAYOUT'});
        result := OBJECT_CONSTRUCT('status', 'SUCCESS', 'file', :file_path);
        RETURN result;
    EXCEPTION
        WHEN OTHER THEN
            error_msg := SQLERRM;
            RETURN OBJECT_CONSTRUCT(
                'status', 'ERROR',
                'file', :file_path,
                'error', :error_msg
            );
    END;
END;
$$;

-- SECTION 11: Monitoring Document AI Usage
-- Track processing statistics
CREATE OR REPLACE TABLE doc_processing_stats (
    stat_date DATE,
    documents_processed INT,
    successful INT,
    failed INT,
    avg_processing_time_sec FLOAT
);

-- SECTION 12: Cleanup (uncomment to run)
-- DROP TASK process_new_documents;
-- DROP STREAM new_documents_stream;
-- DROP TABLE processed_documents;
-- DROP TABLE extracted_documents;
-- DROP TABLE doc_processing_stats;
-- DROP FUNCTION extract_invoice_fields(TEXT);
-- DROP PROCEDURE process_document_safe(TEXT);
-- DROP STAGE document_stage;
-- DROP SCHEMA extraction;
-- DROP DATABASE document_ai_lab;
""",

        "default": """
-- ============================================
-- SQL Lab: Snowflake Feature Demonstration
-- Based on Snowflake Documentation Best Practices
-- ============================================

-- SECTION 1: Environment Setup
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;

-- SECTION 2: Core Feature Demo
-- This lab covers fundamental Snowflake concepts

-- Create sample database and schema
CREATE DATABASE IF NOT EXISTS feature_lab;
USE DATABASE feature_lab;
CREATE SCHEMA IF NOT EXISTS demo;
USE SCHEMA demo;

-- SECTION 3: Sample Table Creation
CREATE OR REPLACE TABLE sample_data (
    id INT AUTOINCREMENT,
    name VARCHAR(100),
    category VARCHAR(50),
    value DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

INSERT INTO sample_data (name, category, value) VALUES
    ('Item A', 'Category 1', 100.00),
    ('Item B', 'Category 2', 200.00),
    ('Item C', 'Category 1', 150.00);

-- SECTION 4: Query and Analysis
SELECT * FROM sample_data;

SELECT 
    category,
    COUNT(*) AS count,
    SUM(value) AS total_value,
    AVG(value) AS avg_value
FROM sample_data
GROUP BY category;

-- SECTION 5: Cleanup (uncomment to run)
-- DROP TABLE sample_data;
-- DROP SCHEMA demo;
-- DROP DATABASE feature_lab;
"""
    }
    
    return labs.get(feature, labs.get("default"))
