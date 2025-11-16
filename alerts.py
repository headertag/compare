import telepot
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_IDS

def initialize_bot():
    """Initializes and returns the Telegram bot object."""
    return telepot.Bot(TELEGRAM_TOKEN)

def send_alert(bot, image_path='ALERT.jpg'):
    """Sends an image alert to the configured Telegram chats."""
    with open(image_path, 'rb') as alert_image:
        for chat_id in TELEGRAM_CHAT_IDS:
            try:
                bot.sendPhoto(chat_id, alert_image)
                alert_image.seek(0)
            except Exception as e:
                print(f"Failed to send photo to {chat_id}: {e}")
