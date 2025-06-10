import asyncio

from dotenv import load_dotenv

from src.llm.business import execute_prompt

load_dotenv()

if __name__ == "__main__":
    print("Hello World")

    response = asyncio.run(execute_prompt("Hello World"))

    print(response)