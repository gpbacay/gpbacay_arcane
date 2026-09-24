"""Synthetic supervision for ARC 1 decision heads.

Generates user utterances paired with the tool calls they should produce,
with exact character spans for every copied argument. Values and some whole
tools are held out so evaluation measures generalisation, not recall:

* ``split="train"`` — training value pools, training tools.
* ``split="eval"``  — held-out values (unseen cities, names, …), training tools.
* ``split="unseen_tools"`` — held-out tools the model never trained on.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .tools import ToolParam, ToolSpec


# ----------------------------------------------------------------- value pools
def _split_pool(values: Sequence[str], eval_frac: float = 0.25) -> Dict[str, List[str]]:
    values = list(values)
    cut = max(1, int(round(len(values) * (1.0 - eval_frac))))
    return {"train": values[:cut], "eval": values[cut:]}


CITIES = _split_pool([
    "Lagos", "Manila", "Tokyo", "Paris", "Berlin", "Sydney", "Cairo", "London", "Toronto",
    "Madrid", "Rome", "Seoul", "Mumbai", "Nairobi", "Lima", "Oslo", "Dubai", "Bangkok",
    "Chicago", "Boston", "Denver", "Austin", "Cebu", "Davao", "Accra", "Kyoto", "Osaka",
    "Vienna", "Prague", "Lisbon", "Dublin", "Athens", "Istanbul", "Hanoi", "Jakarta",
    "Santiago", "Bogota", "Quito", "Havana", "Montreal", "Seattle", "Miami", "Phoenix",
    "Zurich", "Geneva", "Munich", "Hamburg", "Warsaw", "Budapest", "New York", "San Francisco",
    "Los Angeles", "Hong Kong", "Buenos Aires", "Cape Town", "Kuala Lumpur", "Mexico City",
    "Rio de Janeiro", "Tel Aviv", "Abu Dhabi", "Baguio", "Iloilo", "Helsinki", "Stockholm",
    "Copenhagen", "Edinburgh", "Nashville", "Portland", "Atlanta", "Houston",
])
NAMES = _split_pool([
    "Alex", "Sam", "Jordan", "Maya", "Liam", "Olivia", "Noah", "Emma", "Ava", "Lucas",
    "Mia", "Ethan", "Zara", "Kai", "Nia", "Omar", "Priya", "Chen", "Yuki", "Diego",
    "Sofia", "Mateo", "Amara", "Kofi", "Ines", "Luca", "Hana", "Ravi", "Leila", "Tariq",
    "Grace", "Ben", "Chloe", "Daniel", "Elena", "Felix", "Gia", "Hugo", "Isla", "Jack",
    "Kira", "Leo", "Mila", "Nate", "Owen", "Paula", "Quinn", "Rosa", "Theo", "Uma",
    "Vera", "Wade", "Ximena", "Yara", "Zeke", "Mom", "Dad", "Grandma", "Coach Rivera",
    "Dr. Patel", "Aunt May",
])
FULL_NAMES = _split_pool([
    "Alex Kim", "Maya Santos", "Liam Chen", "Olivia Brown", "Noah Garcia", "Emma Wilson",
    "Priya Sharma", "Diego Ramirez", "Sofia Rossi", "Kofi Mensah", "Yuki Tanaka", "Omar Haddad",
    "Grace Lee", "Daniel Cruz", "Elena Petrova", "Felix Wagner", "Hana Sato", "Ravi Patel",
    "Leila Nasser", "Isla Murphy", "Jack Taylor", "Kira Novak", "Leo Martins", "Mila Horvat",
    "Nate Johnson", "Paula Reyes", "Quinn Walsh", "Rosa Delgado", "Theo Laurent", "Uma Iyer",
    "Vera Ivanova", "Wade Carter", "Ximena Lopez", "Yara Silva", "Zeke Owens", "Ana Bautista",
    "Marco Dela Cruz", "Juan Mendoza",
])
ROOMS = _split_pool([
    "living room", "bedroom", "kitchen", "office", "bathroom", "garage", "hallway",
    "dining room", "nursery", "basement", "patio", "guest room", "study", "attic",
])
SONGS = _split_pool([
    "Bohemian Rhapsody", "Blinding Lights", "Shape of You", "Hotel California", "Imagine",
    "Levitating", "Yesterday", "Clair de Lune", "Dancing Queen", "Hey Jude", "Wonderwall",
    "Purple Rain", "Lose Yourself", "Rolling in the Deep", "lo-fi beats", "jazz for studying",
    "Taylor Swift", "the Beatles", "Adele", "Coldplay", "BTS", "classical piano", "Mozart",
    "Billie Eilish", "Bruno Mars", "Queen",
])
EVENTS = _split_pool([
    "team standup", "dentist appointment", "project review", "lunch with Sam", "yoga class",
    "board meeting", "birthday party", "flight to Tokyo", "parent teacher conference",
    "quarterly planning", "coffee chat", "doctor visit", "gym session", "book club",
    "sprint retro", "client demo", "piano lesson", "date night",
])
DATES = _split_pool([
    "tomorrow", "today", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "next Monday", "next week", "March 3", "April 12", "June 21", "July 4",
    "August 30", "September 15", "October 9", "November 2", "December 24", "January 5",
    "the 14th", "2024-05-17", "2025-01-08", "May 1st",
])
TIMES = _split_pool([
    "7am", "7:30 am", "6 pm", "8:15", "noon", "midnight", "9:45 pm", "10 am", "3pm",
    "5:30", "11:00", "6:45 am", "2 pm", "4:20 pm", "13:00", "21:30", "8 o'clock", "half past six",
])
QUERIES = _split_pool([
    "best pizza near me", "how tall is Mount Everest", "python list comprehension",
    "weather radar", "cheap flights to Paris", "how to boil an egg", "latest iPhone reviews",
    "who won the world cup", "symptoms of flu", "tax deadline", "vegan lasagna recipe",
    "meaning of serendipity", "distance to the moon", "open source licenses",
    "how to tie a tie", "tagalog greetings", "history of Rome", "electric car prices",
    "marathon training plan", "black holes explained",
])
ITEMS = _split_pool([
    "pizza", "sushi", "a burger", "pad thai", "tacos", "ramen", "a salad", "fried chicken",
    "adobo", "pho", "dumplings", "a burrito", "pancakes", "lasagna", "biryani", "falafel",
    "sisig", "halo-halo", "shawarma", "poke bowl",
])
LANGUAGES = _split_pool([
    "Spanish", "French", "German", "Japanese", "Tagalog", "Korean", "Italian", "Portuguese",
    "Mandarin", "Hindi", "Arabic", "Russian", "Dutch", "Swahili", "Turkish", "Greek",
])
PHRASES = _split_pool([
    "good morning", "where is the station", "thank you very much", "I love this city",
    "how much does this cost", "see you tomorrow", "happy birthday", "the meeting is at noon",
    "please call me back", "I am running late", "nice to meet you", "where is the bathroom",
])
TICKERS = _split_pool([
    "AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "NVDA", "META", "NFLX", "IBM", "ORCL",
    "AMD", "INTC", "SONY", "BABA", "JPM", "KO", "DIS", "UBER",
])
TASKS = _split_pool([
    "buy milk", "call the plumber", "submit the report", "water the plants", "pay rent",
    "book flights", "renew passport", "clean the garage", "email the landlord",
    "pick up the kids", "walk the dog", "finish the slides", "order printer ink",
    "schedule a haircut", "return the library books", "fix the bike",
])
MESSAGES = _split_pool([
    "dinner is ready", "I'll be there in 10 minutes", "running late", "happy birthday",
    "can you call me back", "the meeting moved to 3pm", "don't forget the tickets",
    "I landed safely", "see you at the gym", "thanks for today", "good luck on the exam",
    "the package arrived", "let's meet at the cafe", "I miss you", "on my way",
])
SUBJECTS = _split_pool([
    "quarterly report", "meeting notes", "invoice 2041", "project update", "vacation request",
    "contract draft", "weekly summary", "budget review", "interview schedule", "bug report",
])
TOPICS = _split_pool([
    "technology", "sports", "the economy", "climate", "politics", "science", "movies",
    "football", "AI", "space", "health", "the stock market", "crypto", "basketball",
])
NOTES = _split_pool([
    "parking spot is B12", "wifi password is on the fridge", "ideas for the trip",
    "gift ideas for mom", "the gate code is 4471", "recipe for banana bread",
    "books to read this summer", "questions for the doctor",
])
FLIGHTS = _split_pool(["PR102", "UA857", "BA283", "JL42", "DL15", "EK203", "5J560", "SQ31", "AA100", "LH400"])
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "PHP", "NGN", "CAD", "AUD", "INR", "KRW", "CHF", "SGD"]
PODCASTS = _split_pool(["The Daily", "Radiolab", "Serial", "Freakonomics", "Hardcore History", "Planet Money", "99% Invisible", "Lex Fridman"])
CUISINES = _split_pool(["Italian", "Thai", "Mexican", "Japanese", "Indian", "Filipino", "Korean", "Greek", "Vietnamese", "Ethiopian"])
MOVIES = _split_pool(["Inception", "Parasite", "Dune", "Up", "Heneral Luna", "Spirited Away", "Arrival", "Coco", "Oppenheimer", "Barbie"])
EMAILS = _split_pool([
    "alex@example.com", "maya.santos@mail.com", "liam.chen@corp.io", "priya@startup.dev",
    "support@shop.ph", "jdoe@uni.edu", "k.mensah@firm.co", "yuki.t@studio.jp",
    "hello@arcane.ai", "orders@store.com", "ravi.p@clinic.org", "info@hotel.ph",
])
PHONES = _split_pool([
    "555-0142", "+63 917 555 0199", "(415) 555-0133", "+44 20 7946 0958", "0917-555-1234",
    "+1 212 555 0101", "555-7788", "+81 3 5555 0100", "+234 803 555 0147", "02-8555-0188",
])
COMPANIES = _split_pool([
    "Acme Corp", "Globex", "Initech", "Umbrella Labs", "Stark Industries", "Wayne Enterprises",
    "Hooli", "Pied Piper", "Jollibee Foods", "Ayala Land", "Northwind", "Contoso", "Tyrell",
])
ORDER_IDS = _split_pool(["A-10293", "ORD-5521", "#88412", "INV-2024-117", "PO-3391", "TX-99812", "R-7730", "SO-1180"])
JOB_TITLES = _split_pool([
    "software engineer", "product manager", "nurse", "teacher", "data scientist",
    "designer", "accountant", "sales lead", "chef", "civil engineer",
])
COUNTRIES = _split_pool(["Philippines", "Nigeria", "Japan", "France", "Germany", "Canada", "Brazil", "Kenya", "India", "Mexico", "Spain", "Vietnam"])


def _int_surface(rng: random.Random, lo: int, hi: int, step: int = 1) -> str:
    return str(rng.randrange(lo, hi + 1, step))


def _amount_surface(rng: random.Random) -> str:
    if rng.random() < 0.6:
        return str(rng.choice([5, 10, 20, 25, 50, 75, 100, 150, 200, 250, 500, 1000, 1200, 2500, 5000]))
    return f"{rng.randint(1, 999)}.{rng.randint(0, 99):02d}"


# ------------------------------------------------------------------ tool defs
@dataclass
class SlotDef:
    """How one parameter is realised in text."""

    sample: Callable[[random.Random, str], str]  # (rng, split) -> surface text
    to_value: Callable[[str], Any] = lambda s: s  # surface -> canonical argument value


@dataclass
class ToolDef:
    name: str
    descriptions: List[str]
    params: List[ToolParam]
    templates: List[str]  # "{param}" placeholders; optional params may be omitted
    slots: Dict[str, SlotDef] = field(default_factory=dict)
    # Literal templates for enum / boolean values that are not copied spans:
    # (template, {param: canonical_value}).
    fixed: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    domain: str = ""

    def spec(self, description: Optional[str] = None) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=description or self.descriptions[0],
            parameters=[ToolParam(p.name, p.type, p.description, p.required, p.enum) for p in self.params],
        )


# ------------------------------------------------------- value synthesis
# Small real pools let the model memorise where "Lagos" ends instead of learning
# where *a value* ends. Half of all training values are re-rolled into a new
# string with the same shape, so span boundaries must come from context.
SYNTH_VALUE_PROB = 0.5
_ONSETS = ["b", "br", "c", "ch", "d", "dr", "f", "g", "gr", "h", "j", "k", "kl", "l", "m", "n", "p", "pr",
           "r", "s", "sh", "st", "t", "tr", "v", "w", "y", "z"]
_VOWELS = ["a", "e", "i", "o", "u", "ai", "ea", "io", "ou"]
_CODAS = ["", "", "", "n", "r", "l", "s", "m", "nd", "rt", "x", "k"]
_KEEP_WORDS = {"a", "an", "the", "of", "to", "for", "in", "on", "at", "my", "is", "and", "with", "from",
               "de", "la", "dr.", "mr.", "ms.", "i", "am", "me", "this", "some"}
COMMON_WORDS = [
    "river", "garden", "silver", "orange", "window", "market", "harbor", "canyon", "meadow", "rocket",
    "pencil", "lantern", "velvet", "copper", "maple", "summit", "bridge", "castle", "violet", "falcon",
    "tiger", "coffee", "planet", "signal", "winter", "autumn", "museum", "island", "forest", "anchor",
    "basket", "candle", "desert", "engine", "fabric", "glacier", "hammer", "jungle", "kettle", "ladder",
    "magnet", "needle", "oyster", "paddle", "quartz", "ribbon", "saddle", "tunnel", "umbrella", "wagon",
    "yellow", "zipper", "blue", "green", "quiet", "happy", "rapid", "golden", "little", "grand",
    "north", "south", "east", "west", "old", "new", "red", "black", "white", "bright",
    "report", "invoice", "photos", "tickets", "groceries", "laundry", "homework", "slides", "budget", "plants",
]


def _pseudo_word(rng: random.Random, length_hint: int) -> str:
    syllables = max(1, min(4, round(length_hint / 3)))
    return "".join(rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS) for _ in range(syllables))


def _match_case(template: str, word: str) -> str:
    if template.isupper() and len(template) > 1:
        return word.upper()
    if template[:1].isupper():
        return word[:1].upper() + word[1:]
    return word.lower()


def synthesize_like(rng: random.Random, value: str) -> str:
    """Re-roll ``value`` keeping its shape: word count, casing, digits, symbols."""
    def repl(m: "re.Match[str]") -> str:
        tok = m.group(0)
        if tok.lower() in _KEEP_WORDS and rng.random() < 0.8:
            return tok
        if tok.isdigit():
            digits = "".join(rng.choice("0123456789") for _ in tok)
            return (rng.choice("123456789") + digits[1:]) if len(tok) > 1 and tok[0] != "0" else digits
        if tok.isalpha():
            if tok.isupper() and len(tok) <= 5:  # tickers, currency-like codes
                return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in tok)
            word = rng.choice(COMMON_WORDS) if rng.random() < 0.4 else _pseudo_word(rng, len(tok))
            return _match_case(tok, word)
        # Mixed alphanumerics (PR102, 5J560): replace char by char.
        out = []
        for ch in tok:
            if ch.isdigit():
                out.append(rng.choice("0123456789"))
            elif ch.isalpha():
                letter = rng.choice("abcdefghijklmnopqrstuvwxyz")
                out.append(letter.upper() if ch.isupper() else letter)
            else:
                out.append(ch)
        return "".join(out)

    out = re.sub(r"[A-Za-z0-9]+", repl, value)
    return out if out.strip() else value


def pool(p: Dict[str, List[str]]) -> Callable[[random.Random, str], str]:
    def sample(rng: random.Random, split: str) -> str:
        value = rng.choice(p["eval" if split == "eval" else "train"] or p["train"])
        if split == "train" and rng.random() < SYNTH_VALUE_PROB:
            value = synthesize_like(rng, value)
        return value
    return sample


def choice_of(values: Sequence[str]) -> Callable[[random.Random, str], str]:
    return lambda rng, split: rng.choice(list(values))


def _num(s: str) -> float:
    return float(re.sub(r"[^0-9.\-]", "", s))


def _int(s: str) -> int:
    return int(float(re.sub(r"[^0-9.\-]", "", s)))


P = ToolParam


def build_tool_library() -> Dict[str, ToolDef]:
    """Training + held-out tools. Descriptions for the four docs-demo tools
    match ``examples/serve_arc1_api.py`` exactly."""
    city = SlotDef(pool(CITIES))
    tools: List[ToolDef] = [
        ToolDef(
            "get_weather",
            ["Get the current weather for a city.", "Look up the weather forecast for a location.", "Returns temperature and conditions for a city."],
            [P("city", "string", "City name")],
            ["what's the weather in {city}?", "weather in {city}", "how's the weather in {city} today", "is it raining in {city}?",
             "forecast for {city} please", "{city} weather", "will it be sunny in {city} tomorrow?", "tell me the temperature in {city}",
             "do I need an umbrella in {city}?", "how hot is it in {city} right now", "check the weather for {city}",
             "what's it like outside in {city}", "is it cold in {city}?"],
            {"city": city}, domain="info",
        ),
        ToolDef(
            "set_lights",
            ["Set light brightness in a room (0-100).", "Set light brightness in a room.", "Dim or brighten the smart lights in a given room."],
            [P("room", "string", "Room name"), P("level", "integer", "Brightness 0-100")],
            ["dim the {room} to {level}", "set the {room} lights to {level}%", "turn the {room} lights to {level} percent",
             "{room} lights at {level}", "make the {room} {level}% bright", "can you set {room} brightness to {level}",
             "lights in the {room} to {level} please", "brighten the {room} to {level}", "set brightness {level} in the {room}",
             "lower the {room} lights to {level}%"],
            {"room": SlotDef(pool(ROOMS)), "level": SlotDef(lambda r, s: _int_surface(r, 0, 100, 5), _int)}, domain="home",
        ),
        ToolDef(
            "convert_currency",
            ["Convert an amount from one currency to another.", "Currency converter using exchange rates.", "Exchange money between two currencies."],
            [P("amount", "number", "Amount to convert"), P("from_currency", "string", "Source currency code, e.g. USD"),
             P("to_currency", "string", "Target currency code, e.g. PHP")],
            ["convert {amount} {from_currency} to {to_currency}", "how much is {amount} {from_currency} in {to_currency}?",
             "{amount} {from_currency} to {to_currency}", "exchange {amount} {from_currency} into {to_currency}",
             "what's {amount} {from_currency} worth in {to_currency}", "change {amount} {from_currency} to {to_currency} please",
             "I have {amount} {from_currency}, how much {to_currency} is that?", "convert {amount} from {from_currency} to {to_currency}"],
            {"amount": SlotDef(lambda r, s: _amount_surface(r), _num),
             "from_currency": SlotDef(lambda r, s: r.choice(CURRENCIES)), "to_currency": SlotDef(lambda r, s: r.choice(CURRENCIES))},
            domain="finance",
        ),
        ToolDef(
            "send_message",
            ["Send a short message to a contact.", "Text a contact through the messaging app.", "Sends an SMS or chat message to someone."],
            [P("to", "string", "Contact name or handle"), P("message", "string", "Message body")],
            ["message {to} that {message}", "text {to}: {message}", "tell {to} {message}", "send {to} a message saying {message}",
             "ping {to} and say {message}", "let {to} know that {message}", "send a text to {to}: {message}",
             "can you message {to} \"{message}\"", "shoot {to} a text that {message}", "msg {to} {message}"],
            {"to": SlotDef(pool(NAMES)), "message": SlotDef(pool(MESSAGES))}, domain="comms",
        ),
        ToolDef(
            "set_alarm",
            ["Set an alarm for a given time.", "Create a wake-up alarm.", "Schedules an alarm clock at a time with an optional label."],
            [P("time", "string", "Alarm time"), P("label", "string", "Optional alarm label", required=False)],
            ["set an alarm for {time}", "wake me up at {time}", "alarm at {time}", "set an alarm at {time} called {label}",
             "I need an alarm for {time} labeled {label}", "wake me at {time} for {label}", "please set my alarm to {time}",
             "can you set an alarm {time}"],
            {"time": SlotDef(pool(TIMES)), "label": SlotDef(choice_of(["gym", "work", "meds", "school run", "flight", "meeting"]))}, domain="time",
        ),
        ToolDef(
            "set_timer",
            ["Start a countdown timer in minutes.", "Set a kitchen timer.", "Starts a timer that rings after N minutes."],
            [P("minutes", "integer", "Duration in minutes")],
            ["set a timer for {minutes} minutes", "timer {minutes} min", "start a {minutes} minute timer", "count down {minutes} minutes",
             "remind me in {minutes} minutes with a timer", "{minutes} minute timer please", "can you time {minutes} minutes"],
            {"minutes": SlotDef(lambda r, s: _int_surface(r, 1, 90), _int)}, domain="time",
        ),
        ToolDef(
            "play_music",
            ["Play a song, artist, or playlist.", "Start music playback.", "Plays music matching a search query."],
            [P("query", "string", "Song, artist, or playlist")],
            ["play {query}", "put on {query}", "I want to hear {query}", "play some {query}", "start playing {query}",
             "can you play {query} on spotify", "queue up {query}", "music: {query}"],
            {"query": SlotDef(pool(SONGS))}, domain="media",
        ),
        ToolDef(
            "create_event",
            ["Create a calendar event.", "Add an event to the calendar on a date.", "Schedules a calendar entry with a title and date."],
            [P("title", "string", "Event title"), P("date", "string", "Event date"), P("time", "string", "Start time", required=False)],
            ["add {title} to my calendar on {date}", "schedule {title} for {date} at {time}", "create an event {title} on {date}",
             "put {title} on {date} in my calendar", "calendar: {title}, {date} at {time}", "book {title} {date}",
             "new event {title} on {date} {time}", "remind my calendar about {title} on {date}"],
            {"title": SlotDef(pool(EVENTS)), "date": SlotDef(pool(DATES)), "time": SlotDef(pool(TIMES))}, domain="time",
        ),
        ToolDef(
            "search_web",
            ["Search the web for a query.", "Look something up online.", "Runs a web search and returns top results."],
            [P("query", "string", "Search query")],
            ["search for {query}", "google {query}", "look up {query}", "find {query} online", "search the web for {query}",
             "can you look up {query}", "web search: {query}"],
            {"query": SlotDef(pool(QUERIES))}, domain="info",
        ),
        ToolDef(
            "book_ride",
            ["Book a ride to a destination.", "Order a taxi or rideshare.", "Requests a car to take the user somewhere."],
            [P("destination", "string", "Where to go"), P("ride_type", "string", "Ride class", required=False, enum=["economy", "premium", "xl"])],
            ["get me a ride to {destination}", "book a car to {destination}", "I need a taxi to {destination}",
             "call an uber to {destination}", "ride to {destination} please", "order a cab to {destination}"],
            {"destination": SlotDef(pool(CITIES))},
            fixed=[("book a premium ride to {destination}", {"ride_type": "premium"}),
                   ("get me an XL car to {destination}", {"ride_type": "xl"}),
                   ("cheapest economy ride to {destination}", {"ride_type": "economy"}),
                   ("I need a big car for 6 people to {destination}", {"ride_type": "xl"}),
                   ("luxury ride to {destination} please", {"ride_type": "premium"}),
                   ("budget ride to {destination}", {"ride_type": "economy"})],
            domain="travel",
        ),
        ToolDef(
            "order_food",
            ["Order food for delivery.", "Place a food delivery order.", "Orders a dish from a delivery app."],
            [P("item", "string", "Dish to order"), P("quantity", "integer", "How many", required=False)],
            ["order {item}", "I want {item} delivered", "get me {item}", "order {quantity} {item}", "can I get {quantity} orders of {item}",
             "deliver {item} please", "I'm hungry, order {item}", "place an order for {quantity} {item}"],
            {"item": SlotDef(pool(ITEMS)), "quantity": SlotDef(lambda r, s: _int_surface(r, 1, 6), _int)}, domain="shopping",
        ),
        ToolDef(
            "translate_text",
            ["Translate text into another language.", "Translate a phrase.", "Machine translation to a target language."],
            [P("text", "string", "Text to translate"), P("target_language", "string", "Language to translate into")],
            ["translate \"{text}\" to {target_language}", "how do you say {text} in {target_language}", "say {text} in {target_language}",
             "translate {text} into {target_language}", "{target_language} for \"{text}\"", "what is {text} in {target_language}?"],
            {"text": SlotDef(pool(PHRASES)), "target_language": SlotDef(pool(LANGUAGES))}, domain="info",
        ),
        ToolDef(
            "set_thermostat",
            ["Set the thermostat temperature.", "Adjust home heating or cooling.", "Changes the target temperature of the HVAC."],
            [P("temperature", "integer", "Target temperature in degrees"), P("mode", "string", "HVAC mode", required=False, enum=["heat", "cool", "auto"])],
            ["set the thermostat to {temperature}", "make it {temperature} degrees", "change the temperature to {temperature}",
             "thermostat {temperature} please", "set the house to {temperature} degrees", "I want it at {temperature}"],
            {"temperature": SlotDef(lambda r, s: _int_surface(r, 16, 30), _int)},
            fixed=[("heat the house to {temperature} degrees", {"mode": "heat"}), ("cool it down to {temperature}", {"mode": "cool"}),
                   ("set the AC to cool at {temperature}", {"mode": "cool"}), ("put the heater on {temperature}", {"mode": "heat"}),
                   ("auto mode at {temperature} degrees", {"mode": "auto"})],
            domain="home",
        ),
        ToolDef(
            "get_stock_price",
            ["Get the latest stock price for a ticker.", "Look up a share price.", "Returns the current market price of a stock."],
            [P("ticker", "string", "Stock ticker symbol")],
            ["what's {ticker} trading at?", "price of {ticker}", "how is {ticker} doing today", "{ticker} stock price",
             "check {ticker} shares", "quote for {ticker}"],
            {"ticker": SlotDef(pool(TICKERS))}, domain="finance",
        ),
        ToolDef(
            "add_todo",
            ["Add a task to the to-do list.", "Create a to-do item.", "Appends an item to the user's task list."],
            [P("task", "string", "Task description")],
            ["add {task} to my to-do list", "remind me to {task}", "todo: {task}", "put {task} on my list", "I need to {task}, add it",
             "new task {task}", "add a to-do to {task}"],
            {"task": SlotDef(pool(TASKS))}, domain="productivity",
        ),
        ToolDef(
            "send_email",
            ["Send an email.", "Compose and send an email to a recipient.", "Emails someone with a subject line."],
            [P("to", "string", "Recipient"), P("subject", "string", "Email subject")],
            ["email {to} about the {subject}", "send {to} an email with subject {subject}", "write an email to {to} re: {subject}",
             "email {to} the {subject}", "shoot {to} an email about {subject}", "compose an email to {to}, subject {subject}"],
            {"to": SlotDef(pool(NAMES)), "subject": SlotDef(pool(SUBJECTS))}, domain="comms",
        ),
        ToolDef(
            "call_contact",
            ["Place a phone call to a contact.", "Call someone.", "Starts a voice call with a contact."],
            [P("name", "string", "Contact to call")],
            ["call {name}", "phone {name}", "ring {name} please", "give {name} a call", "dial {name}", "can you call {name} for me"],
            {"name": SlotDef(pool(NAMES))}, domain="comms",
        ),
        ToolDef(
            "get_directions",
            ["Get directions to a place.", "Navigate to a destination.", "Returns a route to a destination."],
            [P("destination", "string", "Where to go"), P("mode", "string", "Travel mode", required=False, enum=["driving", "walking", "transit"])],
            ["directions to {destination}", "how do I get to {destination}", "navigate to {destination}", "route to {destination}",
             "show me the way to {destination}", "take me to {destination}"],
            {"destination": SlotDef(pool(CITIES))},
            fixed=[("walking directions to {destination}", {"mode": "walking"}), ("how do I drive to {destination}", {"mode": "driving"}),
                   ("take the train to {destination}, show me the route", {"mode": "transit"}),
                   ("bus route to {destination}", {"mode": "transit"}), ("can I walk to {destination}? show me", {"mode": "walking"})],
            domain="travel",
        ),
        ToolDef(
            "lock_door",
            ["Lock a door.", "Lock one of the smart locks.", "Secures a door with the smart lock."],
            [P("door", "string", "Which door", enum=["front", "back", "garage"])],
            [],
            {},
            fixed=[("lock the front door", {"door": "front"}), ("lock the back door", {"door": "back"}),
                   ("lock the garage", {"door": "garage"}), ("did I lock the front? lock it", {"door": "front"}),
                   ("secure the garage door", {"door": "garage"}), ("lock up the back entrance", {"door": "back"}),
                   ("please lock the main entrance", {"door": "front"})],
            domain="home",
        ),
        ToolDef(
            "set_volume",
            ["Set the speaker volume (0-100).", "Change the volume.", "Adjusts media volume level."],
            [P("level", "integer", "Volume 0-100")],
            ["volume {level}", "set the volume to {level}", "turn it up to {level}", "turn the volume down to {level}%",
             "make the speaker {level} percent", "volume at {level} please"],
            {"level": SlotDef(lambda r, s: _int_surface(r, 0, 100, 5), _int)}, domain="media",
        ),
        ToolDef(
            "get_news",
            ["Get the latest news on a topic.", "Fetch news headlines.", "Returns recent headlines for a topic."],
            [P("topic", "string", "News topic")],
            ["news about {topic}", "what's new in {topic}", "latest {topic} headlines", "any {topic} news today?",
             "give me the {topic} news", "headlines on {topic}"],
            {"topic": SlotDef(pool(TOPICS))}, domain="info",
        ),
        ToolDef(
            "create_note",
            ["Save a note.", "Write a quick note.", "Stores a text note for later."],
            [P("content", "string", "Note text")],
            ["note that {content}", "make a note: {content}", "save a note saying {content}", "jot down {content}",
             "write down {content}", "remember this: {content}"],
            {"content": SlotDef(pool(NOTES))}, domain="productivity",
        ),
        ToolDef(
            "check_flight",
            ["Check a flight's status.", "Look up flight status by flight number.", "Returns departure and arrival status for a flight."],
            [P("flight_number", "string", "Flight number")],
            ["is flight {flight_number} on time?", "status of {flight_number}", "check flight {flight_number}",
             "has {flight_number} landed?", "when does {flight_number} depart", "track flight {flight_number}"],
            {"flight_number": SlotDef(pool(FLIGHTS))}, domain="travel",
        ),
        ToolDef(
            "set_reminder",
            ["Set a reminder at a time.", "Remind the user about something.", "Creates a timed reminder."],
            [P("task", "string", "What to be reminded about"), P("time", "string", "When")],
            ["remind me to {task} at {time}", "set a reminder to {task} at {time}", "at {time} remind me to {task}",
             "reminder: {task} at {time}", "don't let me forget to {task} at {time}"],
            {"task": SlotDef(pool(TASKS)), "time": SlotDef(pool(TIMES))}, domain="time",
        ),
        ToolDef(
            "toggle_wifi",
            ["Turn wifi on or off.", "Enable or disable wireless networking.", "Switches the wifi radio."],
            [P("enabled", "boolean", "True to turn wifi on")],
            [], {},
            fixed=[("turn on wifi", {"enabled": True}), ("turn off the wifi", {"enabled": False}), ("enable wifi please", {"enabled": True}),
                   ("disable wifi", {"enabled": False}), ("switch the wifi off", {"enabled": False}), ("wifi on", {"enabled": True}),
                   ("kill the wifi", {"enabled": False}), ("reconnect wifi, turn it on", {"enabled": True})],
            domain="device",
        ),
        ToolDef(
            "set_do_not_disturb",
            ["Turn do not disturb on or off.", "Silence notifications.", "Toggles do-not-disturb mode."],
            [P("enabled", "boolean", "True to enable do not disturb")],
            [], {},
            fixed=[("turn on do not disturb", {"enabled": True}), ("disable do not disturb", {"enabled": False}),
                   ("silence my notifications", {"enabled": True}), ("turn off dnd", {"enabled": False}),
                   ("I'm in a meeting, mute notifications", {"enabled": True}), ("unmute my notifications", {"enabled": False}),
                   ("enable dnd", {"enabled": True})],
            domain="device",
        ),
        # ----------------------------------------------------- held-out tools
        ToolDef(
            "get_time",
            ["Get the current local time in a city.", "Tells the time in a location."],
            [P("city", "string", "City name")],
            ["what time is it in {city}?", "current time in {city}", "time in {city} now", "is it night in {city} yet? what time is it"],
            {"city": city}, domain="info",
        ),
        ToolDef(
            "play_podcast",
            ["Play a podcast by name.", "Start a podcast episode."],
            [P("name", "string", "Podcast name")],
            ["play the {name} podcast", "put on {name}", "start {name} podcast", "latest episode of {name}"],
            {"name": SlotDef(pool(PODCASTS))}, domain="media",
        ),
        ToolDef(
            "find_restaurant",
            ["Find restaurants by cuisine in a city.", "Search for places to eat."],
            [P("cuisine", "string", "Type of food"), P("city", "string", "City")],
            ["find {cuisine} food in {city}", "good {cuisine} restaurants in {city}?", "where can I eat {cuisine} in {city}",
             "{cuisine} places near {city}"],
            {"cuisine": SlotDef(pool(CUISINES)), "city": city}, domain="info",
        ),
        ToolDef(
            "rate_movie",
            ["Rate a movie from 1 to 5 stars.", "Leave a star rating for a film."],
            [P("title", "string", "Movie title"), P("stars", "integer", "Rating 1-5")],
            ["rate {title} {stars} stars", "give {title} {stars} out of 5", "{title} deserves {stars} stars", "I'd rate {title} a {stars}"],
            {"title": SlotDef(pool(MOVIES)), "stars": SlotDef(lambda r, s: _int_surface(r, 1, 5), _int)}, domain="media",
        ),
        ToolDef(
            "toggle_bluetooth",
            ["Turn bluetooth on or off.", "Enable or disable bluetooth."],
            [P("enabled", "boolean", "True to turn bluetooth on")],
            [], {},
            fixed=[("turn on bluetooth", {"enabled": True}), ("switch bluetooth off", {"enabled": False}),
                   ("disable bluetooth", {"enabled": False}), ("enable bluetooth", {"enabled": True})],
            domain="device",
        ),
    ]
    return {t.name: t for t in tools}


HELD_OUT_TOOLS = ("get_time", "play_podcast", "find_restaurant", "rate_movie", "toggle_bluetooth")

CHITCHAT = [
    "tell me a joke", "who are you", "hello there", "thanks!", "good morning", "how are you doing?",
    "what can you do", "that's great", "never mind", "ok cool", "you're awesome", "what's the meaning of life",
    "I'm bored", "lol", "hmm let me think", "can you keep a secret?", "goodnight", "sorry, wrong chat",
    "what is 2 plus 2", "sing me a song", "I had a long day", "do you dream?", "bye", "yes", "no thanks",
]

JOINERS = [" and ", " and also ", ", then ", ". Also ", "; ", " plus "]


# ------------------------------------------------------------------ examples
@dataclass
class Call:
    tool: str
    arguments: Dict[str, Any]
    spans: Dict[str, Tuple[int, int]]  # param -> [char_start, char_end) in user text


@dataclass
class Example:
    user: str
    tools: List[ToolSpec]
    calls: List[Call]
    kind: str = "tool"


def _render(template: str, values: Dict[str, str]) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    """Fill ``{slot}`` placeholders left to right, recording each value's char span."""
    out, spans, pos = [], {}, 0
    length = 0
    for m in re.finditer(r"\{(\w+)\}", template):
        literal = template[pos:m.start()]
        out.append(literal)
        length += len(literal)
        value = values[m.group(1)]
        spans[m.group(1)] = (length, length + len(value))
        out.append(value)
        length += len(value)
        pos = m.end()
    out.append(template[pos:])
    return "".join(out), spans


def _slots_in(template: str) -> List[str]:
    return re.findall(r"\{(\w+)\}", template)


def _surface_noise(rng: random.Random, text: str) -> str:
    """Length-preserving noise (spans stay valid): casing + trailing punctuation."""
    r = rng.random()
    if r < 0.25:
        text = text.lower()
    elif r < 0.35:
        text = text[:1].upper() + text[1:]
    return text


def _single(rng: random.Random, tool: ToolDef, split: str) -> Tuple[str, Call]:
    use_fixed = tool.fixed and (not tool.templates or rng.random() < 0.35)
    if use_fixed:
        template, preset = rng.choice(tool.fixed)
    else:
        template, preset = rng.choice(tool.templates), {}
    surfaces = {name: tool.slots[name].sample(rng, split) for name in _slots_in(template)}
    if tool.name == "convert_currency" and surfaces.get("from_currency") == surfaces.get("to_currency"):
        surfaces["to_currency"] = rng.choice([c for c in CURRENCIES if c != surfaces["from_currency"]])
    text, spans = _render(template, surfaces)
    text = _surface_noise(rng, text)
    args: Dict[str, Any] = dict(preset)
    for name, (a, b) in spans.items():
        args[name] = tool.slots[name].to_value(text[a:b])
    return text, Call(tool.name, args, spans)


def _tool_subset(rng: random.Random, lib: Dict[str, ToolDef], must: Sequence[str], pool_names: Sequence[str],
                 exclude: Sequence[str] = ()) -> List[ToolSpec]:
    k = rng.randint(max(1, len(must)), min(len(must) + 3, len(pool_names)))
    chosen = list(must)
    domains = {lib[m].domain for m in must}
    # Hard negatives: same-domain distractors half the time.
    same = [n for n in pool_names if lib[n].domain in domains and n not in chosen and n not in exclude]
    if same and rng.random() < 0.5:
        chosen.append(rng.choice(same))
    others = [n for n in pool_names if n not in chosen and n not in exclude]
    rng.shuffle(others)
    while len(chosen) < k and others:
        chosen.append(others.pop())
    rng.shuffle(chosen)
    return [lib[n].spec(rng.choice(lib[n].descriptions)) for n in chosen]


# ------------------------------------------------------ procedural tools
# 26 real tools are too few to learn "does this description match this
# request?" in general. Procedural verb/object tools force lexical matching
# between the request and the tool name/description instead of memorising names.
PROCEDURAL_TOOL_PROB = 0.4
_VERBS = {
    "get": ["get", "show", "fetch"], "check": ["check", "look up", "verify"], "find": ["find", "search for", "locate"],
    "track": ["track", "follow", "monitor"], "cancel": ["cancel", "call off", "abort"], "book": ["book", "reserve", "schedule"],
    "start": ["start", "begin", "launch"], "stop": ["stop", "end", "halt"], "open": ["open", "unlock", "launch"],
    "delete": ["delete", "remove", "erase"], "update": ["update", "change", "edit"], "share": ["share", "forward", "send"],
    "renew": ["renew", "extend", "refresh"], "report": ["report", "flag", "log"], "pay": ["pay", "settle", "cover"],
}
_OBJECTS = [
    "package", "parcel", "invoice", "subscription", "reservation", "booking", "playlist", "recipe", "score",
    "balance", "prescription", "ticket", "table", "appointment", "workout", "meeting", "document", "album",
    "delivery", "order", "bill", "parking spot", "library book", "membership", "warranty", "vaccine record",
    "battery", "route", "shift", "lesson", "exam", "loan", "donation", "survey", "podcast", "episode",
    "garden hose", "sprinkler", "doorbell", "camera", "printer", "vacuum", "oven", "car", "bike",
]
_SLOT_KINDS = [
    ("city", "string", "City name", CITIES), ("contact", "string", "Person to involve", NAMES),
    ("code", "string", "Reference code", ORDER_IDS), ("date", "string", "Date", DATES),
    ("time", "string", "Time of day", TIMES), ("item", "string", "Item name", ITEMS),
    ("title", "string", "Title", EVENTS), ("topic", "string", "Topic", TOPICS),
]


def _procedural_tool(rng: random.Random, used_objects: set) -> ToolDef:
    verb = rng.choice(list(_VERBS))
    obj = rng.choice([o for o in _OBJECTS if o not in used_objects] or _OBJECTS)
    used_objects.add(obj)
    slug = obj.replace(" ", "_")
    kinds = rng.sample(_SLOT_KINDS, rng.choice([1, 1, 2]))
    params, slots = [], {}
    for pname, ptype, pdesc, pool_values in kinds:
        params.append(P(pname, ptype, pdesc))
        slots[pname] = SlotDef(pool(pool_values))
    if rng.random() < 0.35:
        params.append(P("count", "integer", "How many", required=False))
        slots["count"] = SlotDef(lambda r, s: _int_surface(r, 1, 20), _int)
    first = kinds[0][0]
    by = " and ".join(k[0] for k in kinds)
    descriptions = [
        f"{verb.capitalize()} a {obj} by {by}.",
        f"{verb.capitalize()} the user's {obj}.",
        f"Use this to {verb} {obj}s for a given {first}.",
    ]
    syn = _VERBS[verb]
    templates = []
    for v in syn:
        templates += [
            f"{v} the {obj} for {{{first}}}", f"please {v} my {obj} {{{first}}}",
            f"can you {v} a {obj} with {{{first}}}", f"I need to {v} the {obj}: {{{first}}}",
        ]
    templates += [f"{obj} for {{{first}}}?", f"what about the {obj} in {{{first}}}", f"{{{first}}} {obj} please"]
    if len(kinds) == 2:
        second = kinds[1][0]
        templates += [f"{v} the {obj} for {{{first}}} and {{{second}}}" for v in syn]
        templates += [f"{v} my {obj}, {first} {{{first}}}, {second} {{{second}}}" for v in syn]
    if "count" in slots:
        templates += [f"{v} {{count}} {obj}s for {{{first}}}" for v in syn]
    return ToolDef(f"{verb}_{slug}", descriptions, params, templates, slots, domain=f"proc_{verb}")


def procedural_library(rng: random.Random, n: int) -> Dict[str, ToolDef]:
    used: set = set()
    tools: Dict[str, ToolDef] = {}
    while len(tools) < n:
        t = _procedural_tool(rng, used)
        tools.setdefault(t.name, t)
    return tools


def sample_tool_example(rng: random.Random, lib: Dict[str, ToolDef], split: str = "train") -> Example:
    names = [n for n in lib if (n in HELD_OUT_TOOLS) == (split == "unseen_tools")]
    pool_names = names
    if split == "train" and rng.random() < PROCEDURAL_TOOL_PROB:
        proc = procedural_library(rng, rng.randint(3, 6))
        lib = {**lib, **proc}
        names = list(proc)                 # the requested tool is procedural...
        pool_names = names + pool_names    # ...distractors mix procedural and real
    return _sample_from(rng, lib, names, pool_names, split)


def _sample_from(rng: random.Random, lib: Dict[str, ToolDef], names: List[str], pool_names: List[str],
                 split: str) -> Example:
    r = rng.random()
    if r < 0.12:  # chit-chat, nothing applies
        text = _surface_noise(rng, rng.choice(CHITCHAT))
        return Example(text, _tool_subset(rng, lib, [], pool_names), [], "none")
    if r < 0.24:  # a real request whose tool is not offered
        tool = lib[rng.choice(names)]
        text, _ = _single(rng, tool, split)
        return Example(text, _tool_subset(rng, lib, [], pool_names, exclude=[tool.name]), [], "none")
    if r < 0.36 and len(names) > 1:  # two requests in one utterance
        a, b = rng.sample(names, 2)
        t1, c1 = _single(rng, lib[a], split)
        t2, c2 = _single(rng, lib[b], split)
        joiner = rng.choice(JOINERS)
        if joiner.startswith(". "):
            t2 = t2[:1].upper() + t2[1:]
            for k, (s, e) in c2.spans.items():  # re-read a value the capital landed on
                if s == 0 and isinstance(c2.arguments.get(k), str):
                    c2.arguments[k] = t2[s:e]
        offset = len(t1) + len(joiner)
        c2.spans = {k: (s + offset, e + offset) for k, (s, e) in c2.spans.items()}
        return Example(t1 + joiner + t2, _tool_subset(rng, lib, [a, b], pool_names), [c1, c2], "compound")
    tool = lib[rng.choice(names)]
    text, call = _single(rng, tool, split)
    return Example(text, _tool_subset(rng, lib, [tool.name], pool_names), [call], "tool")


# --------------------------------------------------------------- extraction
@dataclass
class FieldDef:
    keys: List[str]
    descriptions: List[str]
    type: str
    sample: Callable[[random.Random, str], str]
    snippets: List[str]  # "{v}" placeholder
    to_value: Callable[[str], Any] = lambda s: s


FIELDS: List[FieldDef] = [
    FieldDef(["name", "full_name", "person"], ["Person name", "Full name of the person", "Customer name"], "string", pool(FULL_NAMES),
             ["my name is {v}", "I'm {v}", "Name: {v}", "this is {v}", "{v} here", "signed, {v}", "From: {v}", "the customer is {v}"]),
    FieldDef(["city", "location"], ["City", "City where the person lives", "Location"], "string", pool(CITIES),
             ["I live in {v}", "based in {v}", "City: {v}", "located in {v}", "I'm writing from {v}", "ship to {v}", "we are in {v}"]),
    FieldDef(["email", "email_address"], ["Email address", "Contact email"], "string", pool(EMAILS),
             ["email me at {v}", "Email: {v}", "my email is {v}", "reach me at {v}", "contact: {v}"]),
    FieldDef(["phone", "phone_number"], ["Phone number", "Contact number"], "string", pool(PHONES),
             ["call me at {v}", "Phone: {v}", "my number is {v}", "text {v}", "mobile {v}"]),
    FieldDef(["company", "organization"], ["Company name", "Employer or organization"], "string", pool(COMPANIES),
             ["I work at {v}", "Company: {v}", "on behalf of {v}", "employed by {v}", "{v} is my employer"]),
    FieldDef(["date", "appointment_date"], ["Date", "Requested date"], "string", pool(DATES),
             ["on {v}", "Date: {v}", "the appointment is {v}", "scheduled for {v}", "available {v}"]),
    FieldDef(["amount", "total"], ["Amount of money", "Total amount"], "number", lambda r, s: _amount_surface(r),
             ["the total is ${v}", "Amount: {v}", "I paid {v} dollars", "charged {v}", "invoice for {v} USD"], _num),
    FieldDef(["order_id", "reference"], ["Order ID", "Reference number"], "string", pool(ORDER_IDS),
             ["order {v}", "Order ID: {v}", "reference {v}", "my order number is {v}", "re: {v}"]),
    FieldDef(["age"], ["Age in years", "Person's age"], "integer", lambda r, s: _int_surface(r, 18, 80),
             ["I am {v} years old", "Age: {v}", "aged {v}", "{v} yrs old"], _int),
    FieldDef(["job_title", "role"], ["Job title", "Role or occupation"], "string", pool(JOB_TITLES),
             ["I work as a {v}", "Role: {v}", "I'm a {v}", "position: {v}"]),
    FieldDef(["country"], ["Country", "Country of residence"], "string", pool(COUNTRIES),
             ["from {v}", "Country: {v}", "living in {v}", "citizen of {v}"]),
]

EXTRACT_TOOL_NAME = "extract_record"
EXTRACT_TOOL_DESCRIPTION = "Extract a typed record from the passage."
SNIPPET_JOINERS = [". ", ", ", "\n", "; ", " and "]


@dataclass
class ExtractExample:
    text: str
    schema: Dict[str, Dict[str, str]]
    record: Dict[str, Any]
    spans: Dict[str, Tuple[int, int]]


def sample_extract_example(rng: random.Random, split: str = "train") -> ExtractExample:
    fields = rng.sample(FIELDS, rng.randint(2, 5))
    present = [f for f in fields if rng.random() < 0.8] or fields[:1]
    parts, spans, record, length = [], {}, {}, 0
    keys = {id(f): rng.choice(f.keys) for f in fields}
    order = present[:]
    rng.shuffle(order)
    for i, f in enumerate(order):
        if i:
            joiner = rng.choice(SNIPPET_JOINERS)
            parts.append(joiner)
            length += len(joiner)
        value = f.sample(rng, split)
        snippet = rng.choice(f.snippets)
        text, sp = _render(snippet.replace("{v}", "{value}"), {"value": value})
        if i == 0:
            text = text[:1].upper() + text[1:]
        a, b = sp["value"]
        spans[keys[id(f)]] = (length + a, length + b)
        parts.append(text)
        length += len(text)
    text = "".join(parts) + rng.choice([".", "", "!", " Thanks."])
    for f in present:
        a, b = spans[keys[id(f)]]
        record[keys[id(f)]] = f.to_value(text[a:b])
    schema = {keys[id(f)]: {"type": f.type, "description": rng.choice(f.descriptions)} for f in fields}
    return ExtractExample(text, schema, record, spans)


def extract_tool_spec(schema: Dict[str, Dict[str, str]]) -> ToolSpec:
    return ToolSpec(
        name=EXTRACT_TOOL_NAME,
        description=EXTRACT_TOOL_DESCRIPTION,
        parameters=[
            ToolParam(name=k, type=str(v.get("type", "string")), description=str(v.get("description", k)), required=False)
            for k, v in schema.items()
        ],
    )


def fixed_eval_set(split: str, n: int, seed: int = 1234) -> List[Example]:
    rng = random.Random(seed + {"train": 0, "eval": 1, "unseen_tools": 2}.get(split, 3))
    lib = build_tool_library()
    return [sample_tool_example(rng, lib, split) for _ in range(n)]


def fixed_extract_set(n: int, seed: int = 4321) -> List[ExtractExample]:
    rng = random.Random(seed)
    return [sample_extract_example(rng, "eval") for _ in range(n)]
