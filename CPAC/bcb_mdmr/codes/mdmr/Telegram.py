import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Telegram:
    def __init__(self):
        self.api = os.getenv("TELEGRAM_API_TOKEN")  # Get API token from .env
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")  # Get chat ID from .env

    def send_msg(self, msg):
        url = f"https://api.telegram.org/bot{self.api}/sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': msg
        }
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()  # Raise an error for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error sending message to Telegram: {e}")
            return None