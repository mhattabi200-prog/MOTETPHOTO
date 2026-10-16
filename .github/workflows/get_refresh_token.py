"""يُشغَّل مرة واحدة على جهازك للحصول على YT_REFRESH_TOKEN.

1) نزّل ملف OAuth Client (Desktop app) من Google Cloud وسمّه client_secret.json
   وضعه بجانب هذا الملف.
2) pip install google-auth-oauthlib
3) python get_refresh_token.py
4) سجّل الدخول بحساب قناة يوتيوب ووافق على الصلاحية.
"""
import json

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")

with open("client_secret.json", encoding="utf-8") as f:
    info = json.load(f)
info = info.get("installed") or info.get("web")

print("\n========= انسخ هذه القيم إلى GitHub Secrets =========")
print("YT_CLIENT_ID     =", info["client_id"])
print("YT_CLIENT_SECRET =", info["client_secret"])
print("YT_REFRESH_TOKEN =", creds.refresh_token)
print("=====================================================")
