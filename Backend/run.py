import os
import logging
from app import create_app
from config import Config

app = create_app()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.info("Starting Flask application...")

    # Load configurations from Config or environment variables
    debug_mode = Config.DEBUG
    port = Config.PORT
    host = os.getenv("HOST", "127.0.0.1")  # Default to localhost

    try:
        app.run(debug=debug_mode, host=host, port=port)
    except Exception as e:
        logging.error(f"Failed to start the Flask application: {e}")