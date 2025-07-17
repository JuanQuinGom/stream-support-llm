import asyncio
import os
import json
import redis

from typing import Optional
from twitchio.ext import commands

from src.llm.llm import init_llm_model, execute_graph_prompt


class RedisMessageQueue:
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.processing = False
        self.queue_name = "twitch_message_queue"
        self.processing_queue_name = f"{self.queue_name}:processing"

    async def initialize(self):
        """Initialize Redis connection"""
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD"),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
            socket_connect_timeout=5,
        )
        try:
            data = self.redis.ping()
            if data is True:
                print("Connected to Redis successfully")
            else:
                raise RuntimeError("Failed to connect to Redis")
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            raise

    async def add_message(self, user_id: str, message: str):
        """Add a message to the Redis queue"""
        if not self.redis:
            await self.initialize()

        queue_item = json.dumps({
            "user_id": user_id,
            "message": message,
            "timestamp": asyncio.get_event_loop().time()
        })
        # Correct async list push
        result = self.redis.lpush(self.queue_name, queue_item)
        print(f"Added message to queue. Queue length: {result}")

        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self.process_queue())

        return result

    async def process_queue(self):
        """Process messages from Redis queue"""
        if self.processing or not self.redis:
            return

        self.processing = True
        try:
            while True:
                # Move message to processing queue with BRPOPLPUSH for reliability
                try:
                    queue_item = self.redis.brpoplpush(
                        self.queue_name,
                        self.processing_queue_name,
                        timeout=1
                    )
                except Exception as e:
                    print(f"Redis error: {e}")

                if not queue_item:
                    # No messages, stop processing
                    break

                try:
                    message_data = json.loads(queue_item)
                    user_id = message_data["user_id"]
                    message = message_data["message"]

                    print(f"Processing message from {user_id}: {message}")
                    # Here you would add your actual processing logic
                    # For example:
                    # response = await execute_graph_prompt(user_id, self.graph, message)
                    # await self.send_response(user_id, response)

                    # Remove from processing queue after successful processing

                    self.redis.lrem(self.processing_queue_name, 1, queue_item)
                except json.JSONDecodeError:
                    print("Failed to decode message from queue")
                    self.redis.lrem(self.processing_queue_name, 1, queue_item)
                except Exception as e:
                    print(f"Error processing message: {e}")
                    # Optionally: move message back to main queue or dead letter queue
        finally:
            self.processing = False

    async def close(self):
        """Clean up Redis connection"""
        if self.redis:
            await self.redis.close()


class TwitchBot(commands.Bot):
    def __init__(self):
        super().__init__(
            token=os.getenv("BOT_TOKEN"),
            prefix="!",
            initial_channels=[os.getenv("CHANNEL_NAME")]
        )
        self.graph = init_llm_model()
        self.message_queue = RedisMessageQueue()


    async def event_ready(self):
        """
        Event triggered when the bot establish the connection with Twitch
        """
        print(f"Logged into Twitch as {self.nick}")
        # Initialize Redis connection when bot starts
        await self.message_queue.initialize()


    def process_message(self, id, message):
        """Process the incoming message data"""
        print("Processed message:", message)

        # message = execute_graph_prompt(id, self.graph, message)
        # print("Response from LLM:", message.content)

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
        await self.message_queue.add_message(ctx.author.id, message)
