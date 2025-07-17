import asyncio

from dotenv import load_dotenv
from src.twitch.business import TwitchBot

load_dotenv()

if __name__ == "__main__":
    print("Hello World")
    bot = TwitchBot()
    asyncio.run(bot.start())



    #
    # graph = init_llm_model()
    # print("Graph initialized successfully.")
    #
    #
    # print(execute_graph_prompt(1, graph, "Hi, my name is Bob"))
    # execute_graph_prompt(1, graph, "Write a short poem about cats")
    # execute_graph_prompt(1, graph, "Now do the same but for dogs")
    #
    # # Get summarize of a question
    # execute_graph_prompt(1, graph, "What's my name?")
    # execute_graph_prompt(2, graph, "Who am I?")

