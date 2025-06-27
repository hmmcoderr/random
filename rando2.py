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
