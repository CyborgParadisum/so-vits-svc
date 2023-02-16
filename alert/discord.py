import os

import requests, json


def send_alert(message):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    main_content = {'content': message}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(webhook_url, json.dumps(main_content), headers=headers)
    print(response.content.decode())
