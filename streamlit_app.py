import streamlit as st
import os
import sys

# -----------------------------------------------------------------------------
# 1. Secrets Management
# Streamlit Cloud stores secrets in st.secrets (TOML), but your snowflake_utils.py 
# expects them in os.environ. We map them here automatically.
# -----------------------------------------------------------------------------
try:
    if hasattr(st, "secrets"):
        # Iterate through all top-level secrets
        for key, value in st.secrets.items():
            # If the secret is a string, add it to environment variables
            if isinstance(value, str):
                os.environ[key] = value
            
            # OPTIONAL: If you organize secrets in TOML like [snowflake] section
            # This handles: st.secrets["snowflake"]["account"] -> SNOWFLAKE_ACCOUNT
            if key.lower() == "snowflake" and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    env_key = f"SNOWFLAKE_{sub_key.upper()}"
                    os.environ[env_key] = sub_value
except Exception as e:
    print(f"Note: Could not load Streamlit secrets into environment: {e}")

# -----------------------------------------------------------------------------
# 2. Application Entry Point
# We import 'main' AFTER setting environment variables to ensure any 
# module-level logic has access to the secrets.
# -----------------------------------------------------------------------------
try:
    from main import main
except ImportError:
    # Fallback if running from a different directory structure
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from main import main

if __name__ == "__main__":
    # Execute the main function from main.py
    main()
