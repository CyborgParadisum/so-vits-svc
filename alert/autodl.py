# python脚本示例
import os

import requests
def send_alert(content):
    headers = {"Authorization": os.getenv("AUTODL_TOKEN")}
    resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                         json={
                             "title": "so-vits-svc",
                             "name": "so-vits-svc",
                             "content": content
                         }, headers = headers)
    print(resp.content.decode())
