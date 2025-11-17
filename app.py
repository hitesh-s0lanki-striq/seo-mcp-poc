import os
from dotenv import load_dotenv
from src.ui.app_ui import AppUI

# Load environment variables
load_dotenv(override=True)

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

if __name__ == "__main__":
    app = AppUI()
    app.run()
