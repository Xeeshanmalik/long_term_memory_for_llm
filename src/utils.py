from mem0 import Memory
import os
import logging

# Custom instructions for memory processing
# These aren't being used right now but Mem0 does support adding custom prompting
# for handling memory retrieval and processing.
CUSTOM_INSTRUCTIONS = """
Extract the Following Information:  

- Key Information: Identify and save the most important details.
- Context: Capture the surrounding context to understand the memory's relevance.
- Connections: Note any relationships to other topics or memories.
- Importance: Highlight why this information might be valuable in the future.
- Source: Record where this information came from when applicable.
"""

logger = logging.getLogger(__name__)

def get_mem0_client():
    try:
        # Get LLM provider and configuration
        llm_provider = os.getenv('LLM_PROVIDER')
        llm_api_key = os.getenv('LLM_API_KEY')
        llm_model = os.getenv('LLM_CHOICE')
        embedding_model = os.getenv('EMBEDDING_MODEL_CHOICE')
        
        # Add more detailed debug logging
        # print(f"All environment variables:")
        # print(f"DATABASE_URL from .env: {os.environ.get('DATABASE_URL')}")
        # print(f"DATABASE_URL from getenv: {os.getenv('DATABASE_URL')}")
        
        # Initialize config dictionary
        config = {}
        
        # Configure LLM based on provider
        if llm_provider == 'openai' or llm_provider == 'openrouter' or llm_provider == 'gemini':
            config["llm"] = {
                "provider": llm_provider,
                "config": {
                    "model": llm_model,
                    "temperature": 0.2,
                    "max_tokens": 2000,
                }
            }
            
            # Set API key in environment if not already set
            if llm_api_key and not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = llm_api_key
            
            # For OpenRouter, set the specific API key
            if llm_provider == 'openrouter' and llm_api_key:
                os.environ["OPENROUTER_API_KEY"] = llm_api_key

            if llm_provider == 'gemini' and llm_api_key:
                os.environ["GOOGLE_API_KEY"] = llm_api_key
        

        elif llm_provider == 'ollama':
            config["llm"] = {
                "provider": "ollama",
                "config": {
                    "model": llm_model,
                    "temperature": 0.2,
                    "max_tokens": 2000,
                }
            }
            
            # Set base URL for Ollama if provided
            llm_base_url = os.getenv('LLM_BASE_URL')
            if llm_base_url:
                config["llm"]["config"]["ollama_base_url"] = llm_base_url
        
        # Configure embedder based on provider
        if llm_provider == 'openai':
            config["embedder"] = {
                "provider": "openai",
                "config": {
                    "model": embedding_model or "text-embedding-3-small",
                }
            }
        if llm_provider == 'openai':
            config["embedder"] = {
                "provider": "openai",
                "config": {
                    "model": embedding_model or "text-embedding-3-small",
                }
            }
        if llm_provider == 'gemini':
            config["embedder"] = {
                "provider": "gemini",
                "config": {
                    "model": embedding_model or "models/embedding-001",
                }
            }
        
        
            
            # # Set API key in environment if not already set
            # if llm_api_key and not os.environ.get("OPENAI_API_KEY"):
            #     os.environ["OPENAI_API_KEY"] = llm_api_key
        
        elif llm_provider == 'ollama':
            config["embedder"] = {
                "provider": "ollama",
                "config": {
                    "model": embedding_model or "nomic-embed-text",
                    "embedding_dims": 768  # Default for nomic-embed-text
                }
            }
            
            # Set base URL for Ollama if provided
            embedding_base_url = os.getenv('LLM_BASE_URL')
            if embedding_base_url:
                config["embedder"]["config"]["ollama_base_url"] = embedding_base_url
        
        # Configure Supabase vector store with connection string from environment
        config["vector_store"] = {
            "provider": "supabase",
            "config": {
                "connection_string": os.getenv('DATABASE_URL'),
                "collection_name": "mem0_memories",
                "embedding_model_dims": 1536 if llm_provider == "openai" else 768
            }
        }
        
        # Add debug logging
        # print(f"Vector store config: {config['vector_store']}")
        
        # config["custom_fact_extraction_prompt"] = CUSTOM_INSTRUCTIONS
        
        # Create and return the Memory client
        return Memory.from_config(config)
        
    except Exception as e:
        logger.error(f"Error creating mem0 client: {str(e)}", exc_info=True)
        raise

def test_db_connection():
    try:
        import psycopg2
        # Use the DATABASE_URL from environment variable instead of hardcoding
        database_url = os.getenv('DATABASE_URL')
        logger.debug(f"Attempting to connect to database with URL: {database_url}")
        
        conn = psycopg2.connect(database_url)
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Check if vecs schema exists
        cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'vecs';")
        if not cur.fetchone():
            logger.debug("Creating vecs schema...")
            cur.execute("CREATE SCHEMA IF NOT EXISTS vecs;")
            conn.commit()
            logger.debug("vecs schema created successfully")
        
        cur.close()
        conn.close()
        logger.debug("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False