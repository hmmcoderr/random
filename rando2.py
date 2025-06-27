import subprocess
import json

def prompt_to_sql(user_prompt):
    system_instruction = """
You are a helpful assistant that converts natural language business queries 
into SQL queries. Assume the table is named 'WIP_DATA' and has columns:
Client, Location, WIP_Amount, WIP_Date, Status
Only generate SQL queries. Do not explain.
"""

    full_prompt = f"{system_instruction}\n\nUser: {user_prompt}\nSQL:"
    
    result = subprocess.run(
        ["ollama", "run", "gemma:4b"],
        input=full_prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    output = result.stdout.decode()
    sql = extract_sql(output)
    return sql

def extract_sql(text):
    lines = text.strip().splitlines()
    sql_lines = [line for line in lines if not line.startswith("User:") and line.strip()]
    return "\n".join(sql_lines)

if __name__ == "__main__":
    print("🧠 Ask your WIP-related question:")
    query = input(">> ")
    sql_query = prompt_to_sql(query)
    print("\n📝 Generated SQL:")
    print(sql_query)






import streamlit as st
import subprocess
import pandas as pd

# Config
st.set_page_config(page_title="📊 WIP Assistant", layout="wide")
st.title("📊 WIP Query Assistant")
st.markdown("Ask your question about WIP data. I’ll generate the SQL for you!")

# Dummy metadata for example
TABLE_NAME = "ERPANA_DWH.AGG_efm_oac_wip"
COLUMNS = ["client_name", "location", "wip_amount", "wip_date", "status"]

def generate_sql(question: str):
    system_prompt = f"""
You are a SQL assistant. Generate SQL queries only.
The table is `{TABLE_NAME}` with columns: {', '.join(COLUMNS)}.
Don't explain anything. Just return the query.
"""
    full_prompt = system_prompt + "\n\nUser: " + question + "\nSQL:"

    result = subprocess.run(
        ["ollama", "run", "gemma:4b"],
        input=full_prompt.encode(),
        stdout=subprocess.PIPE
    )
    output = result.stdout.decode()
    return output.strip()

# Input
user_question = st.text_input("Ask a WIP-related question:")
if user_question:
    with st.spinner("Generating SQL..."):
        sql_query = generate_sql(user_question)
    st.subheader("📝 SQL Query")
    st.code(sql_query, language="sql")

    # Optionally: Simulate result with dummy data
    dummy_data = pd.DataFrame({
        "client_name": ["ABC Ltd", "XYZ Inc"],
        "location": ["Delhi", "Mumbai"],
        "wip_amount": [12000, 18000],
        "wip_date": ["2024-04-10", "2024-05-20"],
        "status": ["Pending", "Closed"]
    })
    st.subheader("🔍 Sample Result")
    st.dataframe(dummy_data)
