import os
from dotenv import load_dotenv

load_dotenv()
# Move one directory up from current directory
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '.'))
print(BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, 'final_models')
print(MODELS_DIR)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

GEMINI_MODEL_NAME = "gemini-2.0-flash"

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
