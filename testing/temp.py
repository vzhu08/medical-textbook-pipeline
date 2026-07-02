# pip install requests python-dotenv
import os, time, requests, datetime as dt
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

# set the window you want (UTC)
start = int(dt.datetime(2025, 10, 1, tzinfo=dt.timezone.utc).timestamp())
end   = int(dt.datetime(2025, 10, 27, tzinfo=dt.timezone.utc).timestamp())

resp = requests.get(
    "https://api.openai.com/v1/usage/costs",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        # optionally scope by project:
        # "OpenAI-Project": os.getenv("OPENAI_PROJECT", ""),
    },
    params={"start_time": start, "end_time": end, "granularity": "day"},
    timeout=30,
)
resp.raise_for_status()
data = resp.json()
print("total_usd:", data.get("total_usd"))
# inspect per-day/model breakdowns:
# print(json.dumps(data, indent=2))
