import asyncio
import os
import redis
import json

from twitchio.ext import commands

from src.llm.llm import init_llm_model, execute_graph_prompt


class TwitchBot(commands.Bot):
    def __init__(self):
        super().__init__(
            token=os.getenv("BOT_TOKEN"),
            prefix="!",
            initial_channels=[os.getenv("CHANNEL_NAME")]
        )
        self.graph = init_llm_model()
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.task_queue = "twitch_tasks"

        # Create task processing background task
        self.loop.create_task(self._process_tasks())

    async def event_ready(self):
        """
        Event triggered when the bot establish the connection with Twitch
        """
        print(f"Logged into Twitch as {self.nick}")

    def process_message(self, id, message):
        """Process the incoming message data"""
        print("Processed message:", message)
        message = execute_graph_prompt(id, self.graph, message)
        print("Response from LLM:", message.content)


    async def _process_tasks(self):
        """Async task processor that runs in the bot's event loop"""
        while True:
            try:
                # Use await with async Redis client (see note below)
                _, task_json = await self.redis.blpop(self.task_queue, timeout=1)
                if task_json:
                    task = json.loads(task_json)
                    response = execute_graph_prompt(
                        task["user_id"],
                        self.graph,
                        task["message"]
                    )

                    channel = self.get_channel(task["channel"])
                    if channel:
                        await channel.send(f"@{task['username']} {response.content}")

            except json.JSONDecodeError:
                print("⚠️ Malformed task skipped")
            except Exception as e:
                print(f"❌ Task failed: {str(e)}")
            await asyncio.sleep(0.1)  # Prevents CPU overuse

    @commands.command()
    async def hello(self, ctx: commands.Context):
        # Send a hello back!
        await ctx.send(f'Hello {ctx.author.name}!')

    @commands.command()
    async def shiko(self, ctx: commands.Context):
        # Send a hello back!
        data = ctx.message
        message = data.content
        # Remove the command prefix
        message = message.replace(f"!{ctx.command.name}", "").strip()

        self.process_message(data.author.id, message)