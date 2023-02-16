import logging
import os
# Import WebClient from Python SDK (github.com/slackapi/python-slack-sdk)
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# WebClient instantiates a client that can call API methods
# When using Bolt, you can use either `app.client` or the `client` passed to listeners.
token = os.environ.get("SLACK_TOKEN")
client = WebClient(token=token)
logger = logging.getLogger(__name__)

channel = "ai-alert"
# ID of the channel you want to send the message to
channel_id = channel


def send_alert(content):
    try:
        # Call the chat.postMessage method using the WebClient
        result = client.chat_postMessage(
            channel=channel_id,
            text=content
        )
        logger.info(result)

    except SlackApiError as e:
        logger.error(f"Error posting message: {e}")
