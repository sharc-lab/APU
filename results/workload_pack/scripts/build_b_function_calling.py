"""Family (b): function-calling / structured-extraction subset.

Design choice: the repo has no BFCL (Berkeley Function Calling Leaderboard)
data or harness, and BFCL itself is not a self-contained dataset we can fetch
deterministically here, so we build a small self-contained design "inspired
by" BFCL's item shape: a natural-language user instruction, a fixed list of
available function schemas (name + JSON-schema-style parameters), and a
single correct function call (name + arguments) as the oracle. Grading is an
exact structural match: canonicalize the candidate call's function name and
argument dict (order-independent) and compare to the oracle dict.

A second task_type, "extraction", asks the model to pull fields out of a
short text into a JSON object matching a fixed schema; grading is the same
exact-dict-match rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ITEMS_DIR, count_tokens, rng_for, write_jsonl  # noqa: E402

FAMILY = "function_calling"

FUNCTIONS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "parameters": {"location": "string", "unit": "string (celsius|fahrenheit)"},
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "parameters": {"to": "string (email)", "subject": "string", "body": "string"},
    },
    {
        "name": "create_calendar_event",
        "description": "Create a calendar event.",
        "parameters": {"title": "string", "date": "string (YYYY-MM-DD)", "duration_minutes": "integer"},
    },
    {
        "name": "search_flights",
        "description": "Search for flights between two cities.",
        "parameters": {"origin": "string", "destination": "string", "date": "string (YYYY-MM-DD)"},
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another.",
        "parameters": {"amount": "number", "from_currency": "string", "to_currency": "string"},
    },
    {
        "name": "lookup_stock_price",
        "description": "Look up the latest price for a stock ticker.",
        "parameters": {"ticker": "string"},
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder at a given time.",
        "parameters": {"message": "string", "datetime": "string (ISO 8601)"},
    },
    {
        "name": "translate_text",
        "description": "Translate text into a target language.",
        "parameters": {"text": "string", "target_language": "string"},
    },
]

CITIES = ["Seattle", "Denver", "Austin", "Boston", "Phoenix", "Chicago", "Miami", "Portland"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]
TICKERS = ["NVDA", "AMD", "INTC", "MSFT", "AAPL", "GOOGL", "AMZN", "QCOM"]
LANGUAGES = ["French", "Spanish", "German", "Japanese", "Korean", "Portuguese"]

N_CALL_ITEMS = 60
N_EXTRACTION_ITEMS = 20


def build_call_item(idx: int) -> dict:
    item_id = f"fc_call_{idx:03d}"
    rng = rng_for(item_id)
    fn = rng.choice(FUNCTIONS)
    args: dict = {}
    for pname, ptype in fn["parameters"].items():
        if fn["name"] == "get_weather":
            args = {"location": rng.choice(CITIES), "unit": rng.choice(["celsius", "fahrenheit"])}
            break
        if fn["name"] == "send_email":
            args = {
                "to": f"user{rng.randrange(100, 999)}@example.com",
                "subject": rng.choice(["Status update", "Meeting notes", "Invoice", "Follow-up"]),
                "body": "Please see the attached details.",
            }
            break
        if fn["name"] == "create_calendar_event":
            args = {
                "title": rng.choice(["Sync", "Review", "Planning", "1:1"]),
                "date": f"2026-{rng.randrange(1,13):02d}-{rng.randrange(1,28):02d}",
                "duration_minutes": rng.choice([15, 30, 45, 60]),
            }
            break
        if fn["name"] == "search_flights":
            dest = rng.choice(CITIES)
            origin = rng.choice([c for c in CITIES if c != dest])
            args = {
                "origin": origin,
                "destination": dest,
                "date": f"2026-{rng.randrange(1,13):02d}-{rng.randrange(1,28):02d}",
            }
            break
        if fn["name"] == "convert_currency":
            frm = rng.choice(CURRENCIES)
            to = rng.choice([c for c in CURRENCIES if c != frm])
            args = {"amount": rng.choice([10, 50, 100, 250, 1000]), "from_currency": frm, "to_currency": to}
            break
        if fn["name"] == "lookup_stock_price":
            args = {"ticker": rng.choice(TICKERS)}
            break
        if fn["name"] == "set_reminder":
            args = {
                "message": rng.choice(["Call the vendor", "Submit timesheet", "Renew license"]),
                "datetime": f"2026-{rng.randrange(1,13):02d}-{rng.randrange(1,28):02d}T09:00:00",
            }
            break
        if fn["name"] == "translate_text":
            args = {"text": rng.choice(["Good morning", "Thank you for your help", "See you tomorrow"]),
                     "target_language": rng.choice(LANGUAGES)}
            break

    instruction = f"User request: call the correct function with the correct arguments for: {_nl_for(fn['name'], args)}"
    prompt = _render_prompt(instruction, FUNCTIONS)
    oracle = {"name": fn["name"], "arguments": args}
    return {
        "item_id": item_id,
        "family": FAMILY,
        "task_type": "function_call",
        "difficulty": "medium" if len(args) > 2 else "easy",
        "prompt_tokens": count_tokens(prompt),
        "prompt": prompt,
        "oracle_answer": oracle,
        "grading": {"method": "exact_dict_match"},
    }


def _nl_for(name: str, args: dict) -> str:
    if name == "get_weather":
        return f"what's the weather in {args['location']} in {args['unit']}?"
    if name == "send_email":
        return f"email {args['to']} with subject '{args['subject']}' saying: {args['body']}"
    if name == "create_calendar_event":
        return f"create a {args['duration_minutes']}-minute event '{args['title']}' on {args['date']}"
    if name == "search_flights":
        return f"find flights from {args['origin']} to {args['destination']} on {args['date']}"
    if name == "convert_currency":
        return f"convert {args['amount']} {args['from_currency']} to {args['to_currency']}"
    if name == "lookup_stock_price":
        return f"what's the current price of {args['ticker']}?"
    if name == "set_reminder":
        return f"remind me to '{args['message']}' at {args['datetime']}"
    if name == "translate_text":
        return f"translate '{args['text']}' into {args['target_language']}"
    return "perform the requested action"


def _render_prompt(instruction: str, functions: list[dict]) -> str:
    fn_block = "\n".join(
        f"- {f['name']}({', '.join(f['parameters'].keys())}): {f['description']} "
        f"Parameters: {f['parameters']}"
        for f in functions
    )
    return (
        "You have access to the following functions:\n"
        f"{fn_block}\n\n"
        f"{instruction}\n\n"
        "Respond with a single JSON object of the form "
        '{"name": "<function_name>", "arguments": {...}} and nothing else.'
    )


def build_extraction_item(idx: int) -> dict:
    item_id = f"fc_extract_{idx:03d}"
    rng = rng_for(item_id)
    name = rng.choice(["Alex Rivera", "Jamie Chen", "Morgan Patel", "Taylor Nguyen", "Sam Okafor"])
    company = rng.choice(["Acme Corp", "Initech", "Globex", "Umbrella LLC", "Soylent Inc"])
    role = rng.choice(["VP of Engineering", "Director of Sales", "Lead Analyst", "Product Manager"])
    email = f"{name.split()[0].lower()}.{name.split()[1].lower()}@{company.lower().replace(' ', '')}.com"
    phone = f"({rng.randrange(200,999)}) {rng.randrange(200,999)}-{rng.randrange(1000,9999)}"
    text = (
        f"Hi, this is {name}, {role} at {company}. You can reach me at {email} "
        f"or by phone at {phone}. Looking forward to connecting."
    )
    schema_note = (
        "Extract the following fields as a JSON object with exactly these keys: "
        "name, company, role, email, phone."
    )
    prompt = f"{schema_note}\n\nText:\n\"{text}\"\n\nRespond with only the JSON object."
    oracle = {"name": name, "company": company, "role": role, "email": email, "phone": phone}
    return {
        "item_id": item_id,
        "family": FAMILY,
        "task_type": "extraction",
        "difficulty": "easy",
        "prompt_tokens": count_tokens(prompt),
        "prompt": prompt,
        "oracle_answer": oracle,
        "grading": {"method": "exact_dict_match"},
    }


def build_items() -> list[dict]:
    items = [build_call_item(i) for i in range(N_CALL_ITEMS)]
    items += [build_extraction_item(i) for i in range(N_EXTRACTION_ITEMS)]
    return items


def main():
    items = build_items()
    write_jsonl(ITEMS_DIR / "b_function_calling.jsonl", items)
    print(f"[b] wrote {len(items)} function-calling/extraction items")


if __name__ == "__main__":
    main()
