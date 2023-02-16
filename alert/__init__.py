import os

from alert import autodl, slack, discord
from dotenv import load_dotenv

load_dotenv(os.path.dirname(__file__) + "/../.env")


def send_alert(content):
    autodl.send_alert(content)
    # slack.send_alert(content)
    discord.send_alert(content)
