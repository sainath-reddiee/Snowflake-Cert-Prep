# Snowflake Certification Prep

## Overview

This is a Streamlit-based web application designed to help users prepare for Snowflake SnowPro certifications. The application provides interactive study tools including quiz generation and SQL labs, powered by Snowflake's Cortex AI services.

The app focuses on the SnowPro Core Certification (COF-C02) and covers key exam domains such as Snowflake architecture, security, performance, and data loading/unloading.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit - A Python-based web framework for data applications
- **Styling**: Custom CSS embedded in the Streamlit app for dark theme and branded styling (Snowflake blue gradients)
- **Layout**: Wide layout with expanded sidebar for navigation

### Application Structure
- **Entry Point**: `app.py` serves as the minimal entry point that imports and runs `main.py`
- **Main Application**: `main.py` contains the Streamlit UI logic, page configuration, and rendering
- **Utilities**: `snowflake_utils.py` handles all Snowflake-related operations including database connections and AI generation

### AI Integration
- **Cortex AI**: Uses Snowflake's Cortex AI service for generating quiz questions and SQL labs
- **Model**: Defaults to `mistral-large` for content generation
- **Fallback**: Includes mock generation functions (`generate_mock_quiz`, `generate_mock_sql_lab`) for when Snowflake connection is unavailable

### Data Management
- **Exam Data**: Stored in `exam_data.json` - contains exam definitions, domains, weights, and topics
- **Structure**: JSON-based configuration that defines certification exams and their associated study content

### Connection Management
- Uses Streamlit's `st.connection("snowflake")` for database connectivity
- Credentials expected via `.streamlit/secrets.toml` or environment variables
- Required variables: `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`

## External Dependencies

### Cloud Services
- **Snowflake**: Primary cloud data platform for database connectivity and Cortex AI services
- **Snowflake Cortex**: AI/ML service used for generating certification prep content

### Python Packages
- **Streamlit**: Web application framework
- **snowflake-connector-python** (implied): Required for Snowflake database connectivity

### Configuration Requirements
- Snowflake account with Cortex AI access enabled
- Warehouse with sufficient credits for Cortex operations
- Proper RBAC permissions for the connecting user