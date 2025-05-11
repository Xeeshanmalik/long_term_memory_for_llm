from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
from mem0 import Memory
import asyncio
import json
import os
import logging
import sys
from utils import get_mem0_client, test_db_connection
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import signal

# Set up logging to stderr
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Default user ID for memory operations
DEFAULT_USER_ID = "user"

# Create a dataclass for our application context
@dataclass
class Mem0Context:
    """Context for the Mem0 MCP server."""
    mem0_client: Memory

@asynccontextmanager
async def mem0_lifespan(server: FastMCP) -> AsyncIterator[Mem0Context]:
    """Manages the Mem0 client lifecycle."""
    logger.debug("Entering mem0_lifespan context manager")
    mem0_client = None
    try:
        logger.debug("Creating Mem0 client...")
        mem0_client = get_mem0_client()
        logger.debug("Mem0 client created successfully")
        logger.debug("Yielding Mem0Context...")
        yield Mem0Context(mem0_client=mem0_client)
        logger.debug("Mem0Context yielded successfully")
    except Exception as e:
        logger.error(f"Error in mem0_lifespan: {str(e)}", exc_info=True)
        if mem0_client:
            try:
                # Try to clean up the client
                logger.debug("Attempting to clean up Mem0 client after error...")
                # Add any cleanup code here if needed
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {str(cleanup_error)}", exc_info=True)
        raise
    finally:
        logger.debug("Cleaning up Mem0 client...")
        if mem0_client:
            try:
                # Add any cleanup code here if needed
                pass
            except Exception as cleanup_error:
                logger.error(f"Error during final cleanup: {str(cleanup_error)}", exc_info=True)

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-mem0",
    description="MCP server for long term memory storage and retrieval with Mem0",
    lifespan=mem0_lifespan
)

@mcp.tool()
async def save_memory(ctx: Context, text: str) -> str:
    """Save information to your long-term memory."""
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        messages = [{"role": "user", "content": text}]
        mem0_client.add(messages, user_id=DEFAULT_USER_ID)
        return f"Successfully saved memory: {text[:100]}..." if len(text) > 100 else f"Successfully saved memory: {text}"
    except Exception as e:
        logger.error(f"Error in save_memory: {str(e)}", exc_info=True)
        return f"Error saving memory: {str(e)}"

@mcp.tool()
async def get_all_memories(ctx: Context) -> str:
    """Get all stored memories for the user."""
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        memories = mem0_client.get_all(user_id=DEFAULT_USER_ID)
        if isinstance(memories, dict) and "results" in memories:
            flattened_memories = [memory["memory"] for memory in memories["results"]]
        else:
            flattened_memories = memories
        return json.dumps(flattened_memories, indent=2)
    except Exception as e:
        logger.error(f"Error in get_all_memories: {str(e)}", exc_info=True)
        return f"Error retrieving memories: {str(e)}"

@mcp.tool()
async def search_memories(ctx: Context, query: str, limit: int = 3) -> str:
    """Search memories using semantic search."""
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        memories = mem0_client.search(query, user_id=DEFAULT_USER_ID, limit=limit)
        if isinstance(memories, dict) and "results" in memories:
            flattened_memories = [memory["memory"] for memory in memories["results"]]
        else:
            flattened_memories = memories
        return json.dumps(flattened_memories, indent=2)
    except Exception as e:
        logger.error(f"Error in search_memories: {str(e)}", exc_info=True)
        return f"Error searching memories: {str(e)}"

def handle_exit(signum, frame):
    logger.info(f"Received signal {signum}")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

def check_environment():
    required_vars = ['LLM_PROVIDER', 'LLM_API_KEY', 'LLM_CHOICE', 'EMBEDDING_MODEL_CHOICE']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    return True

async def main():
    try:
        logger.debug("Starting server initialization...")
        
        # Check environment variables
        if not check_environment():
            logger.error("Environment check failed")
            sys.exit(1)
            
        # Test database connection
        if not test_db_connection():
            logger.error("Database connection failed")
            sys.exit(1)
            
        logger.debug("Database connection successful")
        
        # Run the server
        transport = os.getenv("TRANSPORT", "stdio")
        logger.debug(f"Using transport: {transport}")
        
        if transport == 'stdio':
            logger.debug("Starting STDIO transport...")
            try:
                await mcp.run_stdio_async()
            except Exception as e:
                logger.error(f"Error in STDIO transport: {str(e)}", exc_info=True)
                raise
            finally:
                logger.debug("STDIO transport finished")
        else:
            logger.debug("Starting SSE transport...")
            try:
                await mcp.run_sse_async()
            except Exception as e:
                logger.error(f"Error in SSE transport: {str(e)}", exc_info=True)
                raise
            finally:
                logger.debug("SSE transport finished")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logger.debug("Main function finished")

if __name__ == "__main__":
    try:
        logger.debug("Starting application...")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logger.debug("Application finished")
