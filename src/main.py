import asyncio

from dotenv import load_dotenv

from src.llm.business import execute_prompt_summary_memory

load_dotenv()

if __name__ == "__main__":
    print("Hello World")

    # response = asyncio.run(execute_prompt("Hello World"))
    response_spam = asyncio.run(execute_prompt_summary_memory("Your chat’s so empty, it’s giving ‘404 Error: Hype Not Found. Fix it with dogehype dot com @Z0b7Gadd"))
