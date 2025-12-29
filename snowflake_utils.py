"""
Snowflake utility functions for Cortex AI and database connections.
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


def generate_quiz_with_cortex(exam_name: str, topic: str, model: str = "mistral-large") -> Optional[str]:
    """
    Generate quiz questions using Snowflake Cortex COMPLETE function.
    
    Args:
        exam_name: Name of the certification exam
        topic: Topic for the questions
        model: LLM model to use (mistral-large or llama3-70b)
    
    Returns:
        Generated quiz questions as JSON string
    """
    system_prompt = f"""You are a Snowflake Certification expert. Create 5 multiple-choice questions for {exam_name} on the topic of {topic}. 
    
For each question:
- Include one correct answer and three plausible distractors
- Make questions scenario-based and practical
- Focus on real-world application of concepts

Return the response as a valid JSON array with this structure:
[
    {{
        "question": "Question text here",
        "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
        "correct_answer": "A",
        "explanation": "Detailed explanation of why this is correct"
    }}
]

Only return the JSON array, no other text."""

    conn = get_snowflake_connection()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            [
                {{'role': 'system', 'content': '{system_prompt.replace("'", "''")}'}},
                {{'role': 'user', 'content': 'Generate the quiz questions now.'}}
            ],
            {{}}
        ) as response
        """
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        if result is not None:
            return result[0]
        return None
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        return None


def generate_sql_lab(feature: str, model: str = "mistral-large") -> Optional[str]:
    """
    Generate SQL practice scripts using Snowflake Cortex.
    
    Args:
        feature: Snowflake feature to practice
        model: LLM model to use
    
    Returns:
        SQL script with explanations
    """
    system_prompt = f"""You are a Snowflake SQL expert. Create a comprehensive SQL practice lab for the feature: {feature}

Include:
1. Setup commands (CREATE statements)
2. Example data insertion
3. Demonstration queries
4. Cleanup commands

Format the response with clear section headers and explanatory comments.
Make the scripts copy-pasteable and educational."""

    conn = get_snowflake_connection()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            [
                {{'role': 'system', 'content': '{system_prompt.replace("'", "''")}'}},
                {{'role': 'user', 'content': 'Generate the SQL lab now.'}}
            ],
            {{}}
        ) as response
        """
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        if result is not None:
            return result[0]
        return None
    except Exception as e:
        st.error(f"Error generating SQL lab: {str(e)}")
        return None


def generate_mock_quiz(exam_name: str, topic: str) -> List[Dict]:
    """
    Generate mock quiz questions for demo/offline mode.
    Used when Snowflake connection is not available.
    """
    mock_questions = {
        "COF-C02": {
            "default": [
                {
                    "question": "Which Snowflake layer is responsible for query optimization and metadata management?",
                    "options": ["A) Storage Layer", "B) Compute Layer", "C) Cloud Services Layer", "D) Network Layer"],
                    "correct_answer": "C",
                    "explanation": "The Cloud Services Layer handles query optimization, metadata management, authentication, and access control. It's the 'brain' of Snowflake's architecture."
                },
                {
                    "question": "What is the minimum Time Travel retention period in Snowflake?",
                    "options": ["A) 0 days", "B) 1 day", "C) 7 days", "D) 90 days"],
                    "correct_answer": "A",
                    "explanation": "Time Travel can be set to 0 days (disabled) up to 90 days for Enterprise Edition. The default is 1 day for Standard Edition."
                },
                {
                    "question": "Which virtual warehouse size should you start with for initial workload testing?",
                    "options": ["A) X-Small", "B) Small", "C) Medium", "D) Large"],
                    "correct_answer": "A",
                    "explanation": "Best practice is to start with X-Small and scale up based on actual performance needs. This follows Snowflake's pay-per-use model efficiently."
                },
                {
                    "question": "What type of data sharing does Snowflake use that doesn't require data copying?",
                    "options": ["A) ETL Sharing", "B) Secure Direct Data Sharing", "C) FTP Transfer", "D) API Export"],
                    "correct_answer": "B",
                    "explanation": "Secure Direct Data Sharing allows sharing live data without copying. Consumers access the provider's data directly through secure views."
                },
                {
                    "question": "Which caching layer stores query results for reuse across users in the same virtual warehouse?",
                    "options": ["A) Metadata Cache", "B) Result Cache", "C) Local Disk Cache", "D) Memory Cache"],
                    "correct_answer": "B",
                    "explanation": "The Result Cache stores query results for 24 hours and can be reused by any user running the same query, regardless of warehouse."
                }
            ]
        },
        "DEA-C01": {
            "default": [
                {
                    "question": "Which Snowflake feature enables continuous, serverless data loading from cloud storage?",
                    "options": ["A) COPY INTO", "B) Snowpipe", "C) Tasks", "D) Streams"],
                    "correct_answer": "B",
                    "explanation": "Snowpipe provides serverless, continuous data loading with auto-ingest capabilities from cloud storage notifications."
                },
                {
                    "question": "What does a Stream in Snowflake capture?",
                    "options": ["A) Query history", "B) Change Data Capture (CDC)", "C) Error logs", "D) Performance metrics"],
                    "correct_answer": "B",
                    "explanation": "Streams capture Change Data Capture (CDC) information - inserts, updates, and deletes - on tables for downstream processing."
                },
                {
                    "question": "Which object type automatically refreshes based on an underlying query definition?",
                    "options": ["A) View", "B) Materialized View", "C) Dynamic Table", "D) External Table"],
                    "correct_answer": "C",
                    "explanation": "Dynamic Tables automatically refresh based on their query definition and target lag, simplifying pipeline maintenance."
                },
                {
                    "question": "What is the recommended approach for orchestrating dependent tasks in Snowflake?",
                    "options": ["A) Separate schedules", "B) Task DAGs", "C) Manual triggers", "D) External schedulers only"],
                    "correct_answer": "B",
                    "explanation": "Task DAGs (Directed Acyclic Graphs) allow you to define task dependencies and orchestrate complex workflows natively in Snowflake."
                },
                {
                    "question": "Which Snowpark language is NOT natively supported?",
                    "options": ["A) Python", "B) Scala", "C) Java", "D) Ruby"],
                    "correct_answer": "D",
                    "explanation": "Snowpark natively supports Python, Scala, and Java. Ruby is not a supported language for Snowpark development."
                }
            ]
        },
        "ARA-C01": {
            "default": [
                {
                    "question": "What is the primary benefit of Snowflake Organizations?",
                    "options": ["A) Lower costs", "B) Centralized account management", "C) Faster queries", "D) More storage"],
                    "correct_answer": "B",
                    "explanation": "Organizations provide centralized management of multiple Snowflake accounts, enabling cross-account governance and resource sharing."
                },
                {
                    "question": "Which feature enables cross-region disaster recovery in Snowflake?",
                    "options": ["A) Time Travel", "B) Database Replication", "C) Cloning", "D) Data Sharing"],
                    "correct_answer": "B",
                    "explanation": "Database Replication enables cross-region and cross-cloud replication for disaster recovery and business continuity."
                },
                {
                    "question": "What is the recommended approach for private connectivity to Snowflake on AWS?",
                    "options": ["A) VPN", "B) AWS PrivateLink", "C) Direct Connect only", "D) Public internet with encryption"],
                    "correct_answer": "B",
                    "explanation": "AWS PrivateLink provides private connectivity to Snowflake without exposing traffic to the public internet."
                },
                {
                    "question": "Which data architecture pattern does Snowflake's Iceberg Tables support?",
                    "options": ["A) Data Warehouse only", "B) Data Lake only", "C) Data Lakehouse", "D) Data Mart"],
                    "correct_answer": "C",
                    "explanation": "Iceberg Tables enable the Data Lakehouse pattern by combining data lake flexibility with data warehouse capabilities."
                },
                {
                    "question": "What mechanism does Snowflake use for automatic workload scaling?",
                    "options": ["A) Manual resizing", "B) Multi-cluster warehouses", "C) Query queuing", "D) Load balancing"],
                    "correct_answer": "B",
                    "explanation": "Multi-cluster warehouses automatically scale out by adding clusters during high concurrency and scale in during low demand."
                }
            ]
        },
        "GES-C01": {
            "default": [
                {
                    "question": "Which Snowflake Cortex function is used for generating text completions with LLMs?",
                    "options": ["A) SUMMARIZE", "B) COMPLETE", "C) TRANSLATE", "D) EMBED_TEXT"],
                    "correct_answer": "B",
                    "explanation": "COMPLETE is the primary function for generating text completions using LLMs like Mistral and Llama in Snowflake Cortex."
                },
                {
                    "question": "What data type does Snowflake use to store embeddings for vector search?",
                    "options": ["A) ARRAY", "B) VARIANT", "C) VECTOR", "D) OBJECT"],
                    "correct_answer": "C",
                    "explanation": "Snowflake uses the VECTOR data type specifically designed to store and efficiently query embedding vectors."
                },
                {
                    "question": "Which similarity metric is commonly used for semantic search with embeddings?",
                    "options": ["A) Euclidean distance", "B) Manhattan distance", "C) Cosine similarity", "D) Hamming distance"],
                    "correct_answer": "C",
                    "explanation": "Cosine similarity is the standard metric for semantic search as it measures the angle between vectors, capturing semantic similarity."
                },
                {
                    "question": "What is the primary use case for the EMBED_TEXT function?",
                    "options": ["A) Text translation", "B) Creating vector embeddings", "C) Sentiment analysis", "D) Text summarization"],
                    "correct_answer": "B",
                    "explanation": "EMBED_TEXT converts text into numerical vector embeddings that can be used for semantic search and similarity matching."
                },
                {
                    "question": "Which architecture pattern combines retrieval with LLM generation for grounded responses?",
                    "options": ["A) ETL", "B) RAG (Retrieval Augmented Generation)", "C) CDC", "D) ELT"],
                    "correct_answer": "B",
                    "explanation": "RAG (Retrieval Augmented Generation) retrieves relevant context from a knowledge base and uses it to ground LLM responses."
                }
            ]
        }
    }
    
    exam_questions = mock_questions.get(exam_name, mock_questions["COF-C02"])
    return exam_questions.get("default", exam_questions["default"])


def generate_mock_sql_lab(feature: str) -> str:
    """
    Generate mock SQL lab scripts for demo/offline mode.
    """
    labs = {
        "Vector Search": """
-- ===========================================
-- SQL Lab: Setting up Vector Search in Snowflake
-- ===========================================

-- Step 1: Create a database and schema for our vector search lab
CREATE DATABASE IF NOT EXISTS vector_search_lab;
USE DATABASE vector_search_lab;
CREATE SCHEMA IF NOT EXISTS embeddings;
USE SCHEMA embeddings;

-- Step 2: Create a table to store documents with embeddings
CREATE OR REPLACE TABLE documents (
    id INT AUTOINCREMENT,
    title VARCHAR(500),
    content TEXT,
    embedding VECTOR(FLOAT, 768),  -- 768-dimensional embedding
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Step 3: Insert sample documents and generate embeddings
INSERT INTO documents (title, content, embedding)
SELECT 
    'Snowflake Architecture',
    'Snowflake is a cloud-native data platform with a unique multi-cluster shared data architecture.',
    SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', content)
FROM (SELECT 'Snowflake is a cloud-native data platform...' as content);

-- Step 4: Perform vector similarity search
-- Find documents similar to a query
WITH query_embedding AS (
    SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', 
        'How does Snowflake handle data storage?') as query_vec
)
SELECT 
    d.title,
    d.content,
    VECTOR_COSINE_SIMILARITY(d.embedding, q.query_vec) as similarity
FROM documents d, query_embedding q
ORDER BY similarity DESC
LIMIT 5;

-- Step 5: Create a vector search service (if available)
-- CREATE CORTEX SEARCH SERVICE document_search
--     ON documents
--     ATTRIBUTES title
--     WAREHOUSE = 'COMPUTE_WH'
--     TARGET_LAG = '1 hour';

-- Cleanup (uncomment to run)
-- DROP TABLE documents;
-- DROP SCHEMA embeddings;
-- DROP DATABASE vector_search_lab;
""",
        "Streams and Tasks": """
-- ===========================================
-- SQL Lab: Streams and Tasks for CDC Pipelines
-- ===========================================

-- Step 1: Create source and target tables
CREATE OR REPLACE TABLE source_orders (
    order_id INT,
    customer_id INT,
    amount DECIMAL(10,2),
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

CREATE OR REPLACE TABLE processed_orders (
    order_id INT,
    customer_id INT,
    amount DECIMAL(10,2),
    status VARCHAR(50),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Step 2: Create a stream to capture changes
CREATE OR REPLACE STREAM orders_stream ON TABLE source_orders;

-- Step 3: Insert test data to generate stream records
INSERT INTO source_orders (order_id, customer_id, amount, status)
VALUES 
    (1, 101, 150.00, 'pending'),
    (2, 102, 275.50, 'pending'),
    (3, 103, 89.99, 'pending');

-- Step 4: View stream contents (CDC records)
SELECT * FROM orders_stream;

-- Step 5: Create a task to process stream data
CREATE OR REPLACE TASK process_orders_task
    WAREHOUSE = 'COMPUTE_WH'
    SCHEDULE = '1 MINUTE'
    WHEN SYSTEM$STREAM_HAS_DATA('orders_stream')
AS
    INSERT INTO processed_orders (order_id, customer_id, amount, status)
    SELECT order_id, customer_id, amount, 'processed'
    FROM orders_stream
    WHERE METADATA$ACTION = 'INSERT';

-- Step 6: Enable the task
ALTER TASK process_orders_task RESUME;

-- Step 7: Monitor task execution
SELECT * FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY())
WHERE NAME = 'PROCESS_ORDERS_TASK'
ORDER BY SCHEDULED_TIME DESC
LIMIT 10;

-- Cleanup (uncomment to run)
-- ALTER TASK process_orders_task SUSPEND;
-- DROP TASK process_orders_task;
-- DROP STREAM orders_stream;
-- DROP TABLE processed_orders;
-- DROP TABLE source_orders;
""",
        "Dynamic Tables": """
-- ===========================================
-- SQL Lab: Dynamic Tables for Declarative Pipelines
-- ===========================================

-- Step 1: Create source tables
CREATE OR REPLACE TABLE raw_sales (
    sale_id INT,
    product_id INT,
    quantity INT,
    unit_price DECIMAL(10,2),
    sale_date DATE,
    region VARCHAR(50)
);

CREATE OR REPLACE TABLE products (
    product_id INT,
    product_name VARCHAR(200),
    category VARCHAR(100)
);

-- Step 2: Insert sample data
INSERT INTO raw_sales VALUES
    (1, 101, 5, 29.99, '2024-01-15', 'North'),
    (2, 102, 3, 49.99, '2024-01-15', 'South'),
    (3, 101, 2, 29.99, '2024-01-16', 'North'),
    (4, 103, 10, 9.99, '2024-01-16', 'East');

INSERT INTO products VALUES
    (101, 'Widget A', 'Electronics'),
    (102, 'Widget B', 'Electronics'),
    (103, 'Gadget X', 'Accessories');

-- Step 3: Create a Dynamic Table for aggregated sales
CREATE OR REPLACE DYNAMIC TABLE daily_sales_summary
    TARGET_LAG = '1 hour'
    WAREHOUSE = 'COMPUTE_WH'
AS
    SELECT 
        s.sale_date,
        p.category,
        s.region,
        COUNT(*) as num_transactions,
        SUM(s.quantity) as total_units,
        SUM(s.quantity * s.unit_price) as total_revenue
    FROM raw_sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY s.sale_date, p.category, s.region;

-- Step 4: Query the Dynamic Table
SELECT * FROM daily_sales_summary ORDER BY sale_date, category;

-- Step 5: Monitor Dynamic Table refresh
SELECT * FROM TABLE(INFORMATION_SCHEMA.DYNAMIC_TABLE_REFRESH_HISTORY())
WHERE NAME = 'DAILY_SALES_SUMMARY'
ORDER BY REFRESH_START_TIME DESC
LIMIT 5;

-- Step 6: Add more data and observe auto-refresh
INSERT INTO raw_sales VALUES
    (5, 102, 7, 49.99, '2024-01-17', 'West');

-- Cleanup (uncomment to run)
-- DROP DYNAMIC TABLE daily_sales_summary;
-- DROP TABLE products;
-- DROP TABLE raw_sales;
""",
        "Cortex LLM Functions": """
-- ===========================================
-- SQL Lab: Snowflake Cortex LLM Functions
-- ===========================================

-- Step 1: Test basic COMPLETE function
SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'mistral-large',
    'Explain Snowflake virtual warehouses in 2 sentences.'
) as response;

-- Step 2: COMPLETE with system prompt (chat format)
SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'mistral-large',
    [
        {'role': 'system', 'content': 'You are a Snowflake expert. Be concise.'},
        {'role': 'user', 'content': 'What is Time Travel in Snowflake?'}
    ],
    {'temperature': 0.7, 'max_tokens': 200}
) as response;

-- Step 3: SUMMARIZE function
SELECT SNOWFLAKE.CORTEX.SUMMARIZE(
    'Snowflake is a cloud-based data warehousing platform that allows 
    organizations to store, process, and analyze large volumes of data. 
    It uses a unique multi-cluster shared data architecture that separates 
    storage from compute, enabling independent scaling of each layer. 
    This architecture provides significant performance and cost benefits 
    compared to traditional data warehouses.'
) as summary;

-- Step 4: SENTIMENT analysis
SELECT 
    review,
    SNOWFLAKE.CORTEX.SENTIMENT(review) as sentiment_score
FROM (
    SELECT 'Snowflake is amazing! Best data platform ever.' as review
    UNION ALL
    SELECT 'The query performance was disappointing.' as review
    UNION ALL
    SELECT 'It works as expected, nothing special.' as review
);

-- Step 5: TRANSLATE function
SELECT SNOWFLAKE.CORTEX.TRANSLATE(
    'Snowflake enables seamless data sharing across organizations.',
    'en',
    'es'
) as spanish_translation;

-- Step 6: EXTRACT_ANSWER for Q&A
SELECT SNOWFLAKE.CORTEX.EXTRACT_ANSWER(
    'Snowflake was founded in 2012 by Benoit Dageville, Thierry Cruanes, 
    and Marcin Zukowski. The company went public in September 2020 with 
    one of the largest software IPOs in history.',
    'When was Snowflake founded?'
) as answer;

-- Step 7: Practical example - Analyze customer feedback
CREATE OR REPLACE TEMPORARY TABLE customer_feedback (
    feedback_id INT,
    customer_id INT,
    feedback_text TEXT
);

INSERT INTO customer_feedback VALUES
    (1, 101, 'Great product! Fast delivery and excellent quality.'),
    (2, 102, 'Product arrived damaged. Very disappointed.'),
    (3, 103, 'Average experience. Nothing remarkable.');

SELECT 
    feedback_id,
    feedback_text,
    SNOWFLAKE.CORTEX.SENTIMENT(feedback_text) as sentiment,
    SNOWFLAKE.CORTEX.SUMMARIZE(feedback_text) as summary
FROM customer_feedback;

-- Cleanup
DROP TABLE customer_feedback;
""",
        "Data Masking": """
-- ===========================================
-- SQL Lab: Dynamic Data Masking Policies
-- ===========================================

-- Step 1: Create a table with sensitive data
CREATE OR REPLACE TABLE customers (
    customer_id INT,
    full_name VARCHAR(200),
    email VARCHAR(200),
    phone VARCHAR(20),
    ssn VARCHAR(11),
    credit_card VARCHAR(19),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Step 2: Insert sample data
INSERT INTO customers (customer_id, full_name, email, phone, ssn, credit_card)
VALUES 
    (1, 'John Smith', 'john.smith@email.com', '555-123-4567', '123-45-6789', '4111-1111-1111-1111'),
    (2, 'Jane Doe', 'jane.doe@email.com', '555-987-6543', '987-65-4321', '5500-0000-0000-0004'),
    (3, 'Bob Wilson', 'bob.wilson@email.com', '555-456-7890', '456-78-9012', '3400-0000-0000-009');

-- Step 3: Create masking policies
-- Email masking - show only domain
CREATE OR REPLACE MASKING POLICY email_mask AS (val STRING) 
RETURNS STRING ->
    CASE 
        WHEN CURRENT_ROLE() IN ('ADMIN', 'DATA_ADMIN') THEN val
        ELSE CONCAT('***@', SPLIT_PART(val, '@', 2))
    END;

-- SSN masking - show last 4 digits only
CREATE OR REPLACE MASKING POLICY ssn_mask AS (val STRING)
RETURNS STRING ->
    CASE
        WHEN CURRENT_ROLE() IN ('ADMIN', 'DATA_ADMIN') THEN val
        ELSE CONCAT('XXX-XX-', RIGHT(REPLACE(val, '-', ''), 4))
    END;

-- Credit card masking - show last 4 digits
CREATE OR REPLACE MASKING POLICY cc_mask AS (val STRING)
RETURNS STRING ->
    CASE
        WHEN CURRENT_ROLE() IN ('ADMIN', 'FINANCE') THEN val
        ELSE CONCAT('****-****-****-', RIGHT(REPLACE(val, '-', ''), 4))
    END;

-- Step 4: Apply masking policies to columns
ALTER TABLE customers MODIFY COLUMN email SET MASKING POLICY email_mask;
ALTER TABLE customers MODIFY COLUMN ssn SET MASKING POLICY ssn_mask;
ALTER TABLE customers MODIFY COLUMN credit_card SET MASKING POLICY cc_mask;

-- Step 5: Test the masking (results depend on your role)
SELECT * FROM customers;

-- Step 6: View applied policies
SELECT * FROM TABLE(INFORMATION_SCHEMA.POLICY_REFERENCES(
    REF_ENTITY_NAME => 'customers',
    REF_ENTITY_DOMAIN => 'TABLE'
));

-- Cleanup (uncomment to run)
-- ALTER TABLE customers MODIFY COLUMN email UNSET MASKING POLICY;
-- ALTER TABLE customers MODIFY COLUMN ssn UNSET MASKING POLICY;
-- ALTER TABLE customers MODIFY COLUMN credit_card UNSET MASKING POLICY;
-- DROP MASKING POLICY email_mask;
-- DROP MASKING POLICY ssn_mask;
-- DROP MASKING POLICY cc_mask;
-- DROP TABLE customers;
"""
    }
    
    return labs.get(feature, f"""
-- ===========================================
-- SQL Lab: {feature}
-- ===========================================

-- This is a practice lab for: {feature}

-- Step 1: Setup
-- Create necessary objects for practicing {feature}

-- Step 2: Implementation
-- Implement the feature following Snowflake best practices

-- Step 3: Testing
-- Run test queries to verify the implementation

-- Step 4: Cleanup
-- Remove test objects when done

-- For detailed documentation, visit:
-- https://docs.snowflake.com/
""")
