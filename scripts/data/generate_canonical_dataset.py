#!/usr/bin/env python3
"""Generate Phase 9 canonical dataset (300 examples, 9 domains).
Usage: python scripts/data/generate_canonical_dataset.py
"""
import json, random, os

random.seed(42)
examples = []

def add(domain, diff, intent, ctx, answer, constraints=None):
    examples.append({
        "id": f"can_{len(examples)+1:03d}", "domain": domain, "source": "synthetic",
        "difficulty": diff, "user_intent": intent, "context": ctx,
        "ideal_answer": answer, "constraints": constraints or ["concise"],
        "metadata": {"origin_dataset": "synthetic", "license": "MIT", "notes": ""}
    })

def add_mt(diff, intent, turns, constraints=None):
    ex = {
        "id": f"can_{len(examples)+1:03d}", "domain": "multi_turn", "source": "synthetic",
        "difficulty": diff, "user_intent": intent, "context": "",
        "ideal_answer": turns[-1]["assistant"], "constraints": constraints or [],
        "metadata": {"origin_dataset": "synthetic", "license": "MIT", "notes": ""},
        "turns": turns
    }
    examples.append(ex)

# === CODING (33) ===
add("coding","easy","Write is_even(n)","Write a Python function is_even(n) that returns True if n is even.","def is_even(n: int) -> bool:\n    return n % 2 == 0")
add("coding","easy","Reverse a string","Write reverse_str(s) that reverses a string.","def reverse_str(s: str) -> str:\n    return s[::-1]")
add("coding","easy","Sum a list","Write sum_list(nums) returning the sum.","def sum_list(nums: list) -> int:\n    return sum(nums)")
add("coding","easy","Find max value","Write find_max(nums) without using built-in max.","def find_max(nums):\n    m = nums[0]\n    for n in nums[1:]:\n        if n > m: m = n\n    return m")
add("coding","easy","Celsius to Fahrenheit","Write c_to_f(c).","def c_to_f(c: float) -> float:\n    return c * 9/5 + 32")
add("coding","easy","Palindrome check","Write is_palindrome(s) ignoring case and spaces.","def is_palindrome(s: str) -> bool:\n    t = s.replace(' ','').lower()\n    return t == t[::-1]")
add("coding","easy","Greeting function","Write greet(name) returning 'Hello, {name}!'","def greet(name: str) -> str:\n    return f'Hello, {name}!'")
add("coding","easy","List to CSV string","Write to_csv(items) joining with commas.","def to_csv(items: list) -> str:\n    return ', '.join(str(x) for x in items)")
add("coding","easy","Fix missing return","Bug: def double(n): n * 2","def double(n):\n    return n * 2",["fix bug","add return"])
add("coding","easy","Fix off-by-one","Bug: for i in range(len(lst)-1): print(lst[i])","for i in range(len(lst)):\n    print(lst[i])",["fix off-by-one"])
add("coding","medium","Flatten nested list","Write flatten(lst) for arbitrarily nested lists.","def flatten(lst):\n    r = []\n    for x in lst:\n        if isinstance(x, list): r.extend(flatten(x))\n        else: r.append(x)\n    return r")
add("coding","medium","LRU Cache","Write an LRUCache class with get/put, capacity 3.","from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, cap):\n        self.c = OrderedDict()\n        self.cap = cap\n    def get(self, k):\n        if k not in self.c: return -1\n        self.c.move_to_end(k)\n        return self.c[k]\n    def put(self, k, v):\n        if k in self.c: self.c.move_to_end(k)\n        self.c[k] = v\n        if len(self.c) > self.cap: self.c.popitem(last=False)")
add("coding","medium","Parse CSV to dicts","Write parse_csv(text) parsing CSV into list of dicts.","def parse_csv(text):\n    lines = text.strip().split('\\n')\n    h = [x.strip() for x in lines[0].split(',')]\n    return [dict(zip(h,[v.strip() for v in l.split(',')])) for l in lines[1:]]")
add("coding","medium","Binary search","Write binary_search(arr, target) returning index or -1.","def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        m = (lo+hi)//2\n        if arr[m] == target: return m\n        elif arr[m] < target: lo = m+1\n        else: hi = m-1\n    return -1")
add("coding","medium","Merge sorted lists","Write merge_sorted(a, b) merging two sorted lists.","def merge_sorted(a, b):\n    r = []; i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]: r.append(a[i]); i += 1\n        else: r.append(b[j]); j += 1\n    r.extend(a[i:]); r.extend(b[j:])\n    return r")
add("coding","medium","Word frequencies","Write word_freq(text) returning dict of counts.","from collections import Counter\ndef word_freq(text):\n    return dict(Counter(text.lower().split()))")
add("coding","medium","Validate parentheses","Write is_valid_parens(s) checking balanced (), [], {}.","def is_valid_parens(s):\n    st = []; m = {')':'(',']':'[','}':'{'}\n    for c in s:\n        if c in m:\n            if not st or st[-1] != m[c]: return False\n            st.pop()\n        elif c in m.values(): st.append(c)\n    return len(st) == 0")
add("coding","medium","Retry decorator","Write retry(max_attempts=3) decorator.","import functools, time\ndef retry(n=3, d=1):\n    def deco(f):\n        @functools.wraps(f)\n        def w(*a, **k):\n            for i in range(n):\n                try: return f(*a, **k)\n                except Exception as e:\n                    if i == n-1: raise\n                    time.sleep(d)\n        return w\n    return deco")
add("coding","medium","Fibonacci with memo","Write fib(n) with memoization.","from functools import lru_cache\n@lru_cache(None)\ndef fib(n: int) -> int:\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)")
add("coding","medium","Group anagrams","Write group_anagrams(words).","from collections import defaultdict\ndef group_anagrams(words):\n    g = defaultdict(list)\n    for w in words: g[''.join(sorted(w))].append(w)\n    return list(g.values())")
add("coding","medium","Password validator","Write is_valid_password(pw): 8+ chars, upper, lower, digit.","import re\ndef is_valid_password(pw):\n    return len(pw)>=8 and bool(re.search(r'[A-Z]',pw)) and bool(re.search(r'[a-z]',pw)) and bool(re.search(r'\\d',pw))")
add("coding","medium","Explain code","Explain: [x for i,x in enumerate(lst) if i%2==0]","Returns elements at even indices (0, 2, 4, ...). List comprehension filters by enumerate index being even.")
add("coding","medium","Fix race condition","Bug: global count += 1 is not thread-safe.","import threading\nlock = threading.Lock()\ndef increment():\n    global count\n    with lock:\n        count += 1")
add("coding","medium","Queue from stacks","Implement Queue using two lists as stacks.","class Queue:\n    def __init__(self):\n        self.inn = []; self.out = []\n    def enqueue(self, x): self.inn.append(x)\n    def dequeue(self):\n        if not self.out:\n            while self.inn: self.out.append(self.inn.pop())\n        return self.out.pop()")
add("coding","hard","Trie data structure","Implement Trie with insert, search, starts_with.","class Trie:\n    def __init__(self):\n        self.ch = {}; self.end = False\n    def insert(self, w):\n        n = self\n        for c in w:\n            if c not in n.ch: n.ch[c] = Trie()\n            n = n.ch[c]\n        n.end = True\n    def search(self, w):\n        n = self._find(w)\n        return n is not None and n.end\n    def starts_with(self, p): return self._find(p) is not None\n    def _find(self, p):\n        n = self\n        for c in p:\n            if c not in n.ch: return None\n            n = n.ch[c]\n        return n")
add("coding","hard","Token bucket limiter","Implement a TokenBucket rate limiter.","import time\nclass TokenBucket:\n    def __init__(self, rate, cap):\n        self.rate = rate; self.cap = cap; self.tokens = cap; self.t = time.monotonic()\n    def consume(self, n=1):\n        now = time.monotonic()\n        self.tokens = min(self.cap, self.tokens + (now-self.t)*self.rate)\n        self.t = now\n        if self.tokens >= n: self.tokens -= n; return True\n        return False")
add("coding","hard","Expression parser","Write eval_expr(s) for expressions like 2+3*4.","import re, operator as op\ndef eval_expr(s):\n    toks = re.findall(r'\\d+|[+*-]', s)\n    nums = [int(toks[0])]; ops = []; i = 1\n    while i < len(toks):\n        if toks[i] == '*': nums[-1] *= int(toks[i+1])\n        else: nums.append(int(toks[i+1])); ops.append(toks[i])\n        i += 2\n    r = nums[0]\n    for j, o in enumerate(ops):\n        r = (op.add if o=='+' else op.sub)(r, nums[j+1])\n    return r")
add("coding","hard","Quicksort in-place","Write quicksort(arr) in-place.","def quicksort(a, lo=0, hi=None):\n    if hi is None: hi = len(a)-1\n    if lo < hi:\n        p = partition(a, lo, hi)\n        quicksort(a, lo, p-1)\n        quicksort(a, p+1, hi)\n    return a\ndef partition(a, lo, hi):\n    pv = a[hi]; i = lo\n    for j in range(lo, hi):\n        if a[j] < pv: a[i],a[j] = a[j],a[i]; i += 1\n    a[i],a[hi] = a[hi],a[i]\n    return i")
add("coding","medium","Simple HTTP handler","Write a basic HTTP GET handler using http.server.","from http.server import BaseHTTPRequestHandler, HTTPServer\nclass Handler(BaseHTTPRequestHandler):\n    def do_GET(self):\n        self.send_response(200)\n        self.send_header('Content-type','text/plain')\n        self.end_headers()\n        self.wfile.write(b'OK')")
add("coding","medium","Debounce function","Write debounce(fn, delay_ms).","import threading\ndef debounce(fn, delay):\n    timer = None\n    def wrapper(*a, **k):\n        nonlocal timer\n        if timer: timer.cancel()\n        timer = threading.Timer(delay/1000, lambda: fn(*a, **k))\n        timer.start()\n    return wrapper")

# === GAMEFAQ (33) ===
add("gamefaq","easy","Extract locations","In Eldervale, start at the Village Square. Head north to the Forest Path. Continue east to Ironspire Keep.",json.dumps({"locations":["Village Square","Forest Path","Ironspire Keep"]}),["extract locations"])
add("gamefaq","easy","List boss locations","The Crystal Caverns boss is the Frost Wyrm (HP: 3000, weakness: fire). Drops Frost Shard.",json.dumps({"bosses":[{"name":"Frost Wyrm","location":"Crystal Caverns","hp":3000,"weakness":"fire","drops":["Frost Shard"]}]}),["exact boss stats"])
add("gamefaq","easy","Extract NPC info","Merchant Brin in Lakeside Village buys fish (5g) and sells Fishing Rod (20g), Bait Pack (5g).",json.dumps({"npc":{"name":"Merchant Brin","location":"Lakeside Village","buys":[{"item":"fish","price":5}],"sells":[{"item":"Fishing Rod","price":20},{"item":"Bait Pack","price":5}]}}),["include prices"])
add("gamefaq","easy","List enemy weaknesses","In Frozen Wastes: Ice Golem (weak: fire, immune: cold), Frost Wolf (weak: lightning), Crystal Spider (weak: blunt).",json.dumps({"enemies":[{"name":"Ice Golem","weakness":"fire","immune":"cold"},{"name":"Frost Wolf","weakness":"lightning"},{"name":"Crystal Spider","weakness":"blunt"}]}),["exact weaknesses"])
add("gamefaq","easy","Extract save points","Save points in Chapter 3: Forest Camp (after bridge), Mountain Shrine (near waterfall), Castle Gatehouse (before boss).",json.dumps({"save_points":[{"name":"Forest Camp","near":"bridge"},{"name":"Mountain Shrine","near":"waterfall"},{"name":"Castle Gatehouse","near":"boss"}]}),["exact locations"])
add("gamefaq","easy","Extract potion recipes","Health Potion = 2 Red Herbs + 1 Spring Water. Mana Potion = 2 Blue Herbs + 1 Spring Water. Antidote = 1 Green Herb + 1 Charcoal.",json.dumps({"potions":[{"name":"Health Potion","materials":[{"item":"Red Herb","qty":2},{"item":"Spring Water","qty":1}]},{"name":"Mana Potion","materials":[{"item":"Blue Herb","qty":2},{"item":"Spring Water","qty":1}]},{"name":"Antidote","materials":[{"item":"Green Herb","qty":1},{"item":"Charcoal","qty":1}]}]}),["exact quantities"])
add("gamefaq","easy","Extract weapon stats","Weapons: Iron Sword (atk 15, weight 3), Steel Greatsword (atk 28, weight 8), Shadow Dagger (atk 12, weight 1, bonus: stealth).",json.dumps({"weapons":[{"name":"Iron Sword","atk":15,"weight":3},{"name":"Steel Greatsword","atk":28,"weight":8},{"name":"Shadow Dagger","atk":12,"weight":1,"bonus":"stealth"}]}),["exact stats"])
add("gamefaq","easy","Extract shop inventory","Mountain Pass Trader: Fire Arrows (10g, bundle of 10), Ice Shield (150g, def +20), Climbing Rope (30g).",json.dumps({"shop":"Mountain Pass Trader","sells":[{"item":"Fire Arrows","price":10,"qty":10},{"item":"Ice Shield","price":150,"def":20},{"item":"Climbing Rope","price":30}]}),["include prices"])
add("gamefaq","medium","Build quest progression","Quest: The Lost Crown. 1: Talk to Elder Theron (Oakhaven). 2: Find Ancient Map (Ruined Library). 3: Navigate Whispering Woods. 4: Defeat Shadow Guardian. 5: Return Crown to Theron.",json.dumps({"quest":"The Lost Crown","steps":[{"step":1,"action":"Talk to Elder Theron","location":"Oakhaven"},{"step":2,"action":"Find Ancient Map","location":"Ruined Library"},{"step":3,"action":"Navigate Whispering Woods"},{"step":4,"action":"Defeat Shadow Guardian","type":"combat"},{"step":5,"action":"Return Crown","location":"Oakhaven"}]}),["preserve order"])
add("gamefaq","medium","Map dungeon layout","Darkhollow: Entrance -> Hall of Echoes (west: Crypt of Bones, east: Flooded Passage). Crypt has a lever opening Flooded Passage gate. Flooded Passage -> Boss Chamber.",json.dumps({"rooms":[{"name":"Entrance","to":["Hall of Echoes"]},{"name":"Hall of Echoes","to":["Crypt of Bones","Flooded Passage"]},{"name":"Crypt of Bones","features":["lever"]},{"name":"Flooded Passage","to":["Boss Chamber"],"gate":"opened by Crypt lever"}]}),["preserve connections"])
add("gamefaq","medium","Extract crafting recipes","Iron Sword = 2 Iron Ingots + 1 Leather. Steel Greatsword = 3 Steel Ingots + 2 Leather + 1 Ruby. Shadow Dagger = 1 Dark Steel + 1 Spider Silk.",json.dumps({"recipes":[{"result":"Iron Sword","materials":[{"item":"Iron Ingot","qty":2},{"item":"Leather","qty":1}]},{"result":"Steel Greatsword","materials":[{"item":"Steel Ingot","qty":3},{"item":"Leather","qty":2},{"item":"Ruby","qty":1}]},{"result":"Shadow Dagger","materials":[{"item":"Dark Steel","qty":1},{"item":"Spider Silk","qty":1}]}]}),["exact quantities"])
add("gamefaq","medium","Build area graph","Overgrown Path -> Mossy Glade, Spider Nest. Mossy Glade -> Fairy Ring (needs Charm). Spider Nest -> Queen's Lair. Fairy Ring -> Sky Garden.",json.dumps({"areas":["Overgrown Path","Mossy Glade","Spider Nest","Fairy Ring","Queen's Lair","Sky Garden"],"edges":[{"from":"Overgrown Path","to":"Mossy Glade"},{"from":"Overgrown Path","to":"Spider Nest"},{"from":"Mossy Glade","to":"Fairy Ring","req":"Charm spell"},{"from":"Spider Nest","to":"Queen's Lair"},{"from":"Fairy Ring","to":"Sky Garden","type":"teleport"}]}),["valid edges"])
add("gamefaq","medium","Extract skill tree","Warrior: T1: Slash(5AP), Block(3AP). T2(requires Slash): Whirlwind(8AP), Cleave(6AP). T3(requires Whirlwind): Berserker Rage(15AP), Earthquake(12AP).",json.dumps({"skills":[{"name":"Slash","tier":1,"cost":"5AP"},{"name":"Block","tier":1,"cost":"3AP"},{"name":"Whirlwind","tier":2,"cost":"8AP","requires":"Slash"},{"name":"Cleave","tier":2,"cost":"6AP","requires":"Slash"},{"name":"Berserker Rage","tier":3,"cost":"15AP","requires":"Whirlwind"},{"name":"Earthquake","tier":3,"cost":"12AP","requires":"Whirlwind"}]}),["preserve tiers"])
add("gamefaq","medium","Extract puzzle solution","Runic Door: 4 slots. Clue: 'Fire binds water, earth opposes air, the first is last, the last is first.' Solution: Air, Earth, Water, Fire.",json.dumps({"puzzle":"Runic Door","slots":4,"solution":["Air","Earth","Water","Fire"],"logic":"reverse with clue"}),["correct solution"])
add("gamefaq","medium","Extract side quest rewards","Missing Miner: Find Borin in Deep Tunnels. Rewards: 200 XP, Dwarven Pickaxe (mining +50%), Gem Cavern shortcut.",json.dumps({"quest":"Missing Miner","objective":"Find Borin","location":"Deep Tunnels","rewards":[{"type":"xp","amount":200},{"item":"Dwarven Pickaxe","bonus":"mining +50%"},{"unlock":"Gem Cavern shortcut"}]}),["all rewards"])
add("gamefaq","medium","Extract treasure locations","Sunken Temple: Room 1: chest behind pillar (200g). Room 3: secret wall left (Enchanted Amulet). Room 5: underwater alcove (Pearl of Wisdom).",json.dumps({"treasures":[{"room":1,"location":"behind pillar","item":"200 gold"},{"room":3,"location":"secret wall left","item":"Enchanted Amulet"},{"room":5,"location":"underwater alcove","item":"Pearl of Wisdom"}]}),["preserve room numbers"])
add("gamefaq","medium","Extract NPC schedule","Blacksmith Gareth: Mon-Fri 6am-12pm Forge (buys/sells), 12-2pm Tavern, 2-6pm Forge, 6-10pm Home. Sat 8am-4pm Market. Sun Chapel.",json.dumps({"npc":"Blacksmith Gareth","schedule":{"weekdays":[{"time":"6-12","location":"Forge"},{"time":"12-2","location":"Tavern"},{"time":"2-6","location":"Forge"},{"time":"6-10","location":"Home"}],"saturday":[{"time":"8-4","location":"Market"}],"sunday":[{"time":"all day","location":"Chapel"}]}}),["full schedule"])
add("gamefaq","medium","Extract map coordinates","POIs: Village(0,0), Dark Forest(3,5), Crystal Lake(-2,7), Volcano(8,3), Ruins(5,-4), Port(-6,2).",json.dumps({"pois":[{"name":"Village","x":0,"y":0},{"name":"Dark Forest","x":3,"y":5},{"name":"Crystal Lake","x":-2,"y":7},{"name":"Volcano","x":8,"y":3},{"name":"Ruins","x":5,"y":-4},{"name":"Port","x":-6,"y":2}]}),["exact coordinates"])
add("gamefaq","medium","Extract boss guide","Lich King (Crypt of Shadows). P1(100-50%): summons 4 skeletons/30s, melee drains 10% mana, weak holy. P2(50-0%): Shadow Bolt(500dmg), summons Mage, invulnerable during summon(interrupt with Silence).",json.dumps({"boss":"Lich King","location":"Crypt of Shadows","phases":[{"phase":1,"hp":"100-50%","abilities":["summon 4 skeletons/30s","mana drain 10%"],"weakness":"holy"},{"phase":2,"hp":"50-0%","abilities":["Shadow Bolt 500dmg","summon Mage (interrupt Silence)"]}]}),["preserve phases"])
add("gamefaq","hard","Full area with secrets","Whispering Library: Main Hall(3 bookshelves). East Wing: symbol puzzle(5 symbols) -> vault(Tome of Knowledge +10INT, 500g). Secret: push 3rd bookshelf -> Hidden Chamber(Librarian Ghost, quest 'Forgotten Archives'). West Wing: Reading Room(save), Archive(Torn Page).",json.dumps({"area":"Whispering Library","rooms":[{"name":"Main Hall","features":"3 bookshelves"},{"name":"East Wing","puzzle":"5 symbols","vault":["Tome of Knowledge +10INT","500g"]},{"name":"West Wing","rooms":["Reading Room(save)","Archive(Torn Page)"]}],"secret":{"trigger":"push 3rd bookshelf","reveals":"Hidden Chamber","npc":"Librarian Ghost","quest":"Forgotten Archives"}}),["include secrets"])
add("gamefaq","hard","Quest dependency graph","Lost Key(no prereq)->Sealed Door. Wolf Hunt(no prereq)->Alpha Wolf, Wolf Pelts. Sealed Door+Alpha Wolf->Ancient Guardian.",json.dumps({"quests":[{"name":"Lost Key","prereqs":[]},{"name":"Sealed Door","prereqs":["Lost Key"]},{"name":"Wolf Hunt","prereqs":[]},{"name":"Alpha Wolf","prereqs":["Wolf Hunt"]},{"name":"Wolf Pelts","prereqs":["Wolf Hunt"]},{"name":"Ancient Guardian","prereqs":["Sealed Door","Alpha Wolf"]}]}),["valid graph"])
add("gamefaq","hard","Full fortress extraction","Thornwall: L1: Armory(rusty key from Guard Captain), Barracks(3 enemies), Stairwell. L2: War Room(Commander Vex, quest 'Siege Breaker'), Prison(captured scout), Stairwell. L3: Throne Room(boss Warlord Kael HP5000, 2 phases, drops Kael's Signet).",json.dumps({"fortress":"Thornwall","levels":[{"level":1,"rooms":[{"name":"Armory","key":"rusty key","key_from":"Guard Captain"},{"name":"Barracks","enemies":3},{"name":"Stairwell"}]},{"level":2,"rooms":[{"name":"War Room","npc":"Commander Vex","quest":"Siege Breaker"},{"name":"Prison","npc":"captured scout","optional":True},{"name":"Stairwell"}]},{"level":3,"rooms":[{"name":"Throne Room","boss":{"name":"Warlord Kael","hp":5000,"phases":2,"drops":["Kael's Signet"]}}]}]}),["preserve hierarchy"])
add("gamefaq","medium","Extract world map","World map regions: Ashenvale(forest, level 1-5), Ironspire Wastes(desert, level 8-12), Crystal Peaks(mountains, level 15-20), Shadowfen(swamp, level 22-28), Dragonspire(volcano, level 30+).",json.dumps({"regions":[{"name":"Ashenvale","biome":"forest","level_range":"1-5"},{"name":"Ironspire Wastes","biome":"desert","level_range":"8-12"},{"name":"Crystal Peaks","biome":"mountains","level_range":"15-20"},{"name":"Shadowfen","biome":"swamp","level_range":"22-28"},{"name":"Dragonspire","biome":"volcano","level_range":"30+"}]}),["preserve level ranges"])
add("gamefaq","easy","Extract merchant prices","Alchemist Elara: Health Potion 15g, Mana Potion 20g, Antidote 8g, Fire Resist Potion 35g. Buys herbs at 3g each.",json.dumps({"npc":"Alchemist Elara","sells":[{"item":"Health Potion","price":15},{"item":"Mana Potion","price":20},{"item":"Antidote","price":8},{"item":"Fire Resist Potion","price":35}],"buys":[{"item":"herbs","price":3}]}),["exact prices"])
add("gamefaq","medium","Extract quest chain","Dragon Slayer chain: 1. 'Egg Hunt'(find 3 dragon eggs in Ashenvale). 2. 'Hatchling'(protect egg for 5 days). 3. 'First Flight'(tame young dragon). 4. 'Dragon Rider'(defeat Dragon Lord). Rewards escalate: 100g, 500g, Dragon Mount, Dragonbane Sword.",json.dumps({"chain":"Dragon Slayer","quests":[{"name":"Egg Hunt","task":"find 3 dragon eggs","location":"Ashenvale"},{"name":"Hatchling","task":"protect egg 5 days"},{"name":"First Flight","task":"tame young dragon"},{"name":"Dragon Rider","task":"defeat Dragon Lord"}],"rewards":["100g","500g","Dragon Mount","Dragonbane Sword"]}),["preserve order","include rewards"])
add("gamefaq","hard","Extract complete walkthrough","Chapter 5: Start at Harbor. Talk to Captain Mira -> get Ship Pass. Use Ship Pass at dock -> sail to Storm Isle. Storm Isle: Navigate lightning maze (solution: right, left, straight, right). Find Storm Crystal in cave. Return to Harbor. Give Crystal to Artificer Zeph -> enchant weapon. Boss: Storm Elemental (weak: earth, immune: lightning, HP 4000).",json.dumps({"chapter":5,"steps":[{"action":"Talk to Captain Mira","location":"Harbor","item":"Ship Pass"},{"action":"Sail to Storm Isle","use":"Ship Pass"},{"action":"Navigate lightning maze","solution":"right, left, straight, right"},{"action":"Find Storm Crystal","location":"cave"},{"action":"Return to Harbor"},{"action":"Give Crystal to Artificer Zeph","result":"enchant weapon"},{"action":"Defeat Storm Elemental","boss":{"weakness":"earth","immune":"lightning","hp":4000}}]}),["preserve all steps","exact boss stats"])

# === JSON (33) ===
add("json","easy","Extract person info","John Smith, 34, Manchester, software engineer at TechCorp.",json.dumps({"name":"John Smith","age":34,"city":"Manchester","occupation":"software engineer","employer":"TechCorp"}),["exact fields"])
add("json","easy","Parse product listing","Product: Wireless Mouse, Brand: Logitech, Price: 29.99 GBP, In Stock, Rating: 4.5/5",json.dumps({"product":"Wireless Mouse","brand":"Logitech","price":{"amount":29.99,"currency":"GBP"},"in_stock":True,"rating":4.5}),["correct types"])
add("json","easy","Extract event details","Conference on March 15 2026 at ICC Birmingham. Registration 8:30am, keynote 9am. Tickets 150 GBP.",json.dumps({"event":"conference","date":"2026-03-15","venue":"ICC Birmingham","schedule":[{"time":"08:30","activity":"registration"},{"time":"09:00","activity":"keynote"}],"price":{"amount":150,"currency":"GBP"}}),["ISO date","exact times"])
add("json","easy","Extract date and time","Meeting: Tuesday 14 Jan 2026 at 2:30 PM GMT.",json.dumps({"day":"Tuesday","date":"2026-01-14","time":"14:30","timezone":"GMT"}),["ISO format"])
add("json","easy","Parse simple key-value","user: name=Sarah, age=28, hobbies=[reading, cycling, coding]",json.dumps({"user":{"name":"Sarah","age":28,"hobbies":["reading","cycling","coding"]}}),["correct types"])
add("json","medium","Parse restaurant","Golden Dragon, Chinese, 42 High St. Mon-Sat 11:30-22:00, Sun 12-21. Avg meal 18 GBP. Vegetarian options. Phone 0161 234 5678.",json.dumps({"name":"Golden Dragon","cuisine":"Chinese","address":"42 High St","hours":{"mon_sat":"11:30-22:00","sunday":"12:00-21:00"},"avg_meal":{"amount":18,"currency":"GBP"},"vegetarian":True,"phone":"0161 234 5678"}),["structured hours"])
add("json","medium","Parse messy address","Flat 3b, 147 St. Georges Road, Salford, Greater Manchester, M3 6DN, UK",json.dumps({"apartment":"3b","number":147,"street":"St. Georges Road","city":"Salford","county":"Greater Manchester","postcode":"M3 6DN","country":"UK"}),["preserve abbreviations"])
add("json","medium","Extract API response","status 200, user_id 8834, username @devops_bill, created 2024-11-03T14:22:00Z, roles [admin,editor], last_login null.",json.dumps({"status":200,"body":{"user_id":8834,"username":"@devops_bill","created_at":"2024-11-03T14:22:00Z","roles":["admin","editor"],"last_login":None}}),["preserve null"])
add("json","medium","Extract weather","Manchester: High 14C, Low 8C, Humidity 72%, Wind 15mph NW, UV 3, Overcast, rain at 3pm.",json.dumps({"city":"Manchester","temp":{"high_c":14,"low_c":8},"humidity":72,"wind":{"mph":15,"dir":"NW"},"uv":3,"conditions":"overcast","forecast":"rain at 3pm"}),["structured temp/wind"])
add("json","medium","Extract movie","Inception (2010), dir Nolan. Cast: DiCaprio, Gordon-Levitt, Page. 148 min. IMDB 8.8. Sci-Fi/Action/Thriller.",json.dumps({"title":"Inception","year":2010,"director":"Christopher Nolan","cast":["Leonardo DiCaprio","Joseph Gordon-Levitt","Elliot Page"],"runtime_min":148,"imdb":8.8,"genres":["Sci-Fi","Action","Thriller"]}),["exact fields"])
add("json","medium","Extract email metadata","From alice@ex.com to bob@ex.com, cc carol@ex.com. Subject: Q4 Report. Date: 5 Jan 2026 09:15 +0000. Priority High.",json.dumps({"from":"alice@example.com","to":["bob@example.com"],"cc":["carol@example.com"],"subject":"Q4 Report","date":"2026-01-05T09:15:00+00:00","priority":"High"}),["ISO date"])
add("json","medium","Extract sports stats","Liverpool 2-1 Arsenal. Goals: Salah 23', Nunez 67', Saka 45'. Possession 58-42%. Shots 14-8. Corners 7-3.",json.dumps({"home":"Liverpool","away":"Arsenal","score":"2-1","goals":[{"scorer":"Salah","min":23},{"scorer":"Nunez","min":67},{"scorer":"Saka","min":45}],"possession":{"Liverpool":58,"Arsenal":42},"shots":{"Liverpool":14,"Arsenal":8}}),["exact stats"])
add("json","medium","Extract flight info","BA2490 LHR->DXB. Dep 2026-04-10 21:35. Arr 2026-04-11 07:45. 6h10m. Boeing 777-300ER. Economy. Seat 32A.",json.dumps({"flight":"BA2490","from":{"code":"LHR"},"to":{"code":"DXB"},"depart":"2026-04-10T21:35","arrive":"2026-04-11T07:45","duration":"6h10m","aircraft":"Boeing 777-300ER","class":"Economy","seat":"32A"}),["ISO datetimes"])
add("json","medium","Extract invoice","Invoice INV-2026-042, 1 Mar 2026, due 31 Mar. Client Acme Corp. Items: Web Dev 40hrs@75, Hosting 1mo@120, Domain@15. Subtotal 3135, VAT 20%=627, Total 3762 GBP.",json.dumps({"invoice":"INV-2026-042","date":"2026-03-01","due":"2026-03-31","client":"Acme Corp","items":[{"desc":"Web Dev","qty":40,"rate":75,"total":3000},{"desc":"Hosting","qty":1,"rate":120,"total":120},{"desc":"Domain","qty":1,"rate":15,"total":15}],"subtotal":3135,"vat":627,"total":3762,"currency":"GBP"}),["all line items","correct totals"])
add("json","medium","Extract package.json","name my-app, version 2.1.0, scripts: start node index.js, test jest --coverage, deps: express ^4.18.2, react ^18.2.0, devDeps: jest ^29.7.0",json.dumps({"name":"my-app","version":"2.1.0","scripts":{"start":"node index.js","test":"jest --coverage"},"dependencies":{"express":"^4.18.2","react":"^18.2.0"},"devDependencies":{"jest":"^29.7.0"}}),["preserve versions"])
add("json","hard","Convert table to JSON","ID|Name|Dept|Salary\nE001|Alice Brown|Engineering|65000\nE002|Bob Wilson|Marketing|52000\nE003|Carol Davis|Engineering|71000",json.dumps({"employees":[{"id":"E001","name":"Alice Brown","dept":"Engineering","salary":65000},{"id":"E002","name":"Bob Wilson","dept":"Marketing","salary":52000},{"id":"E003","name":"Carol Davis","dept":"Engineering","salary":71000}]}),["preserve all rows","correct types"])
add("json","hard","Extract error log","2026-01-15 14:32:01 ERROR [api-gateway] Connection refused to db-primary:5432 after 3 retries. Stack: /app/db.py line 45 connect. Context: user_service.read_user(8834). Impact: 503 to client 192.168.1.45.",json.dumps({"timestamp":"2026-01-15T14:32:01","level":"ERROR","service":"api-gateway","error":"Connection refused","target":"db-primary:5432","retries":3,"stack":{"file":"/app/db.py","line":45},"context":"user_service.read_user(8834)","impact":"503 to 192.168.1.45"}),["all fields"])
add("json","hard","Extract complex schema","User types: Admin(read,write,delete,manage_users), Editor(read,write), Viewer(read). Fields: id(UUID), email(unique), name, type(enum), created_at(ISO8601), last_login(nullable), settings(theme:string,notifications:bool,language:string).",json.dumps({"user_types":{"Admin":["read","write","delete","manage_users"],"Editor":["read","write"],"Viewer":["read"]},"fields":{"id":"UUID","email":{"type":"string","unique":True},"name":"string","type":{"enum":["Admin","Editor","Viewer"]},"created_at":"ISO8601","last_login":{"type":"ISO8601","nullable":True},"settings":{"theme":"string","notifications":"bool","language":"string"}}}),["preserve schema"])
add("json","hard","Extract nested company","TechNova: Engineering(Sarah Chen, 45 staff, teams: Backend/Mike Torres, Frontend/Lisa Kim, DevOps/Tom Hughes), Marketing(James Wright, 12), Operations(Priya Patel, 8).",json.dumps({"company":"TechNova","departments":[{"name":"Engineering","head":"Sarah Chen","count":45,"teams":[{"name":"Backend","lead":"Mike Torres"},{"name":"Frontend","lead":"Lisa Kim"},{"name":"DevOps","lead":"Tom Hughes"}]},{"name":"Marketing","head":"James Wright","count":12},{"name":"Operations","head":"Priya Patel","count":8}]}),["preserve hierarchy"])
add("json","easy","Extract color palette","Primary: #1a73e8, Secondary: #34a853, Error: #ea4335, Background: #ffffff, Text: #202124.",json.dumps({"primary":"#1a73e8","secondary":"#34a853","error":"#ea4335","background":"#ffffff","text":"#202124"}),["exact hex values"])
add("json","medium","Extract git log","Commits: a3b2c1d 'Fix login bug' by Alice 2026-01-10. d4e5f6a 'Add dark mode' by Bob 2026-01-12. b7c8d9e 'Refactor API' by Carol 2026-01-14.",json.dumps({"commits":[{"hash":"a3b2c1d","message":"Fix login bug","author":"Alice","date":"2026-01-10"},{"hash":"d4e5f6a","message":"Add dark mode","author":"Bob","date":"2026-01-12"},{"hash":"b7c8d9e","message":"Refactor API","author":"Carol","date":"2026-01-14"}]}),["exact hashes"])
add("json","medium","Extract Docker compose","services: web (image: nginx:1.25, port 80:80), api (build: ./api, port 3000:3000, env: DB_HOST=postgres), postgres (image: postgres:16, volume: pgdata:/var/lib/postgresql/data).",json.dumps({"services":{"web":{"image":"nginx:1.25","ports":["80:80"]},"api":{"build":"./api","ports":["3000:3000"],"env":{"DB_HOST":"postgres"}},"postgres":{"image":"postgres:16","volumes":["pgdata:/var/lib/postgresql/data"]}}}),["preserve structure"])
add("json","medium","Extract test results","Test suite: 42 passed, 3 failed, 1 skipped. Failed: test_login_timeout (AssertionError, line 87), test_api_rate_limit (TimeoutError), test_export_format (ValueError). Coverage: 87.3%.",json.dumps({"total":46,"passed":42,"failed":3,"skipped":1,"failures":[{"name":"test_login_timeout","error":"AssertionError","line":87},{"name":"test_api_rate_limit","error":"TimeoutError"},{"name":"test_export_format","error":"ValueError"}],"coverage":87.3}),["exact counts","include failures"])

# === REASONING (33) ===
reasoning = [
    ("easy","Train at 60mph for 2.5 hours, distance?","","Distance = 60 * 2.5 = 150 miles."),
    ("easy","Apples at 1.20 GBP each, cost for 7?","","7 * 1.20 = 8.40 GBP."),
    ("easy","Today is Wednesday, what day in 10 days?","","10 mod 7 = 3. Wednesday + 3 = Saturday."),
    ("easy","15% of 240?","","0.15 * 240 = 36."),
    ("easy","Next in sequence: 2, 6, 12, 20, 30, ?","","Differences: 4,6,8,10,12. Next: 30+12 = 42."),
    ("medium","Chickens and cows: 30 heads, 80 legs. How many?","Let c=chickens, w=cows. c+w=30, 2c+4w=80.","c=20 chickens, w=10 cows."),
    ("medium","Bill 87.50 GBP, 15% tip, 3 people split?","","87.50*1.15 = 100.625. Each: 33.54 GBP."),
    ("medium","Ball dropped 16m, bounces half. Height after 4 bounces?","","8, 4, 2, 1. After 4 bounces: 1 meter."),
    ("medium","20% off then additional 10% off. Total discount?","","0.80 * 0.90 = 0.72. Total discount: 28%."),
    ("medium","5 machines make 5 widgets in 5 min. 100 machines, 100 widgets?","","Each machine makes 1 widget in 5 min. 100 machines: 5 minutes."),
    ("medium","Pipe A: 6hrs, Pipe B: 4hrs. Together?","","1/6 + 1/4 = 5/12 per hour. Time: 2.4 hours."),
    ("medium","Class 30: 18 football, 15 cricket, 8 both. Neither?","","18+15-8 = 25. Neither: 30-25 = 5."),
    ("medium","Bill splits: 87.50 + 15% tip, 3 ways.","Each person's share with tip included.","Total: 100.63 GBP. Each: 33.54 GBP."),
    ("medium","Average speed: 40mph first half, 60mph second half (by distance)?","","Harmonic mean: 2*40*60/(40+60) = 48 mph."),
    ("hard","Clock at 3:15, angle between hands?","","Minute: 90 deg. Hour: 97.5 deg. Angle: 7.5 degrees."),
    ("hard","12 balls, 1 heavier. Min weighings?","","3 weighings. Split 4-4-4, then narrow down."),
    ("hard","Birthday paradox with 23 people?","","P = 1 - prod(365-i)/365 for i=0..22 = ~50.7%."),
    ("hard","5 people in a row, Alice next to Bob, Charlie not at ends.","Treat Alice+Bob as unit, constrain Charlie.","24 valid arrangements."),
    ("medium","Water jug: 3L and 5L, measure exactly 4L?","","Fill 5L, pour into 3L (2L left in 5L). Empty 3L. Pour 2L into 3L. Fill 5L. Pour into 3L (only 1L fits). 5L now has 4L."),
    ("easy","If all roses are flowers, and some flowers fade quickly, do all roses fade?","","No. 'Some flowers' doesn't mean all flowers. We can't conclude all roses fade."),
    ("medium","A farmer doubles his crop each year. Starting with 1 plant, after how many years will he have over 1000?","","2^10 = 1024. After 10 years."),
    ("medium","How many squares on a chessboard?","","Not just 64. Count all sizes: 64(1x1) + 49(2x2) + ... + 1(8x8) = 204."),
    ("hard","Monty Hall problem: should you switch?","","Yes. Switching wins 2/3 of the time. Initial pick has 1/3 chance; the revealed door concentrates 2/3 on the remaining."),
    ("medium","3 red, 2 blue balls. Draw 2 without replacement. P(both red)?","","(3/5)*(2/4) = 6/20 = 3/10 = 30%."),
    ("easy","A pizza is cut into 8 slices. You eat 3. What fraction remains?","","5/8 remains."),
    ("medium","Speed of sound is 343 m/s. How far is lightning if thunder takes 3 seconds?","","343 * 3 = 1029 meters, about 1 km."),
    ("hard","100 doors problem: toggle every nth. Which are open at the end?","","Perfect squares: 1,4,9,16,25,36,49,64,81,100. Only they have an odd number of factors."),
    ("medium","How many times do clock hands overlap in 24 hours?","","22 times. About every 65.45 minutes."),
    ("easy","A dozen eggs costs 3.60 GBP. Cost per egg?","","3.60 / 12 = 0.30 GBP per egg."),
    ("medium","You have 100 coins: 99 fair, 1 double-headed. Pick random coin, flip heads. P(other side is heads)?","Bayes theorem.","P(double|heads) = P(heads|double)*P(double) / P(heads) = 1*0.01 / (0.01 + 0.99*0.5) = 0.01/0.505 = ~1.98%."),
    ("hard","Prisoner's dilemma iterated 100 rounds, both rational. What happens?","","Backward induction: both defect every round (Nash equilibrium). Cooperation unravels from the last round."),
    ("medium","If it takes 5 minutes to boil 1 egg, how long to boil 5 eggs?","","Still 5 minutes (boil them together in the same pot)."),
    ("hard","Two envelopes paradox: one has twice the other. Should you switch?","","No net gain from switching. The paradox arises from incorrectly mixing prior and posterior probabilities."),
]
for diff,intent,ctx,answer in reasoning:
    add("reasoning",diff,intent,ctx,answer)

# === GENERAL (33) ===
general = [
    ("easy","Store leftover pasta","How to store leftover cooked pasta?","Airtight container in fridge, 3-5 days. Toss with oil to prevent sticking."),
    ("easy","Remove coffee stain","How to remove coffee stain from white shirt?","Blot (don't rub). Apply vinegar + dish soap mix 1:1. Wait 15 min. Wash cold."),
    ("easy","USB-A vs USB-C difference","What's the difference?","USB-A: rectangular, older. USB-C: oval, reversible, faster (40Gbps), more power (240W), carries video."),
    ("easy","Car oil change frequency","How often change car oil?","Every 5,000-7,500 miles (synthetic) or once a year. Check your manual."),
    ("easy","Phone battery tips","How to make phone battery last?","Reduce brightness, disable background refresh, use Wi-Fi over mobile data, enable low power mode at 20%."),
    ("easy","Peel garlic fast","Best way to peel garlic?","Crush clove with flat side of knife. Skin slides off."),
    ("easy","Stretching for desk workers","Quick stretch routine?","Every hour: neck rolls, shoulder shrugs, wrist circles, spinal twist, stand and touch toes. 3 minutes."),
    ("easy","Boil eggs time","How long to soft-boil an egg?","6 minutes in boiling water, then ice bath immediately."),
    ("medium","Home network security","How to secure home network?","Change default password, use WPA3, update firmware, set up guest network, disable WPS, enable firewall."),
    ("medium","Tech interview prep","How to prepare for tech interview?","Research company + stack. Review job description line by line. Practice coding. Prepare questions to ask. Test video setup."),
    ("medium","Buying used laptop","What to check?","Battery health, screen pixels, keyboard, ports, SSD health, RAM, warranty. Test under load 15min."),
    ("medium","Pour-over coffee","How to make good pour-over?","15g coffee per 250ml, medium-fine grind. Water 93-96C. Bloom 30ml/30s. Pour in circles. 3-4 min total."),
    ("medium","Start investing small budget","How to begin investing?","Open stocks ISA (UK, 20k/yr tax-free). Start with low-cost global index fund. Invest regularly. Keep 3-6 months cash first."),
    ("medium","Learn new programming language","Fastest way to learn?","Pick a small project. Read just enough syntax. Build it, googling as you go. Read others' code. Repeat with harder project."),
    ("medium","Renting vs buying","Key differences?","Renting: flexible, no maintenance, no equity. Buying: builds equity, stable payments, responsible for repairs, ties you to location."),
    ("medium","Automated backups","How to set up backups?","Linux: rsync + cron. Mac: Time Machine. Windows: File History. Test restoring monthly. Keep one offsite backup."),
    ("medium","Identity theft response","What to do if identity stolen?","Freeze bank accounts. Report Action Fraud. Check credit report. Change passwords. Contact HMRC. Get crime reference number."),
    ("medium","Prepare for power outage","How to prepare?","Charge devices, fill water containers, stock canned food + torch + batteries, fill car with fuel, cash in hand."),
    ("medium","Set up home office ergonomics","Ergonomic setup?","Monitor at eye level, arm's length away. Elbows at 90 deg. Feet flat on floor. Chair supports lumbar. Take breaks every hour."),
    ("medium","Cook rice perfectly","How to cook rice?","Rinse until water runs clear. 1:1.5 rice to water ratio. Bring to boil, simmer covered 18 min. Rest 5 min. Don't peek."),
    ("medium","Reduce food waste","How to reduce food waste?","Plan meals, make shopping list, store food properly, use leftovers creatively, freeze before expiry, compost scraps."),
    ("medium","Improve sleep quality","How to sleep better?","Same bedtime/wake time. No screens 1hr before. Cool dark room. No caffeine after 2pm. Exercise earlier in the day."),
    ("easy","Remove red wine stain","How to remove red wine stain?","Blot immediately. Apply salt liberally. After 5 min, rinse with cold water. Apply dish soap + hydrogen peroxide mix."),
    ("easy","Unclog a drain naturally","How to unclog without chemicals?","Pour 1/2 cup baking soda, then 1/2 cup vinegar. Cover 30 min. Flush with boiling water."),
    ("hard","Explain HTTPS to non-technical","How does HTTPS work (simply)?","Browser and server do a secret handshake. Server proves identity via certificate. They agree on a secret code. All data is scrambled with that code. The padlock icon means this worked."),
    ("hard","Explain blockchain simply","What is blockchain (simple)?","A shared digital ledger that everyone can see but no one can cheat. Each page (block) is linked to the previous one with a math lock. Changing one page would break all the locks after it."),
    ("hard","Explain machine learning simply","What is ML (simple)?","Showing a computer thousands of examples until it spots patterns. Like teaching a child to recognise cats by showing many cat photos, rather than writing rules about whiskers and tails."),
    ("easy","Remove sticker residue","How to get sticker residue off glass?","Rub with cooking oil or WD-40, wait 5 min, scrape with credit card. Wipe clean."),
    ("medium","Fix slow computer","Computer running slow, what to check?","Check Task Manager for high-CPU apps. Clear temp files. Check startup programs. Run disk cleanup. Check for malware. Consider SSD upgrade."),
    ("medium","Choose a password manager","Which password manager?","Bitwarden (free tier, open source) or 1Password (polished, family plans). Enable 2FA. Generate unique passwords for every site."),
    ("medium","Set up a VPN","How to use a VPN?","Choose a reputable provider (Mullvad, ProtonVPN). Install their app. Connect to nearest server. Enable kill switch. Test for DNS leaks at ipleak.net."),
    ("medium","Explain DNS simply","What is DNS?","The internet's phone book. When you type google.com, DNS translates it to an IP address (like 142.250.187.206) so your browser knows where to connect."),
    ("easy","Convert miles to km","Convert 5 miles to km.","5 * 1.609 = 8.045 km."),
]
for diff,intent,ctx,answer in general:
    add("general",diff,intent,ctx,answer)

# === DEOBFUSCATION (33) ===
deobf = [
    ("easy","Rename variables","def f(a,b): return a+b", "def add(x, y):\n    return x + y","Original used single-letter names. Renamed for clarity."),
    ("easy","Rename variables","x = [i for i in range(10) if i % 2 == 0]","even_numbers = [num for num in range(10) if num % 2 == 0]","List comprehension generating even numbers 0-8."),
    ("easy","Explain obfuscated code","def p(s): print(s[::-1])","Reverses and prints a string. p('hello') prints 'olleh'. The [::-1] slice reverses."),
    ("easy","Rename variables","a1 = 10; b1 = 20; c1 = a1 + b1","base_price = 10\ntax_rate = 20\ntotal = base_price + tax_rate","Simple addition, likely price calculation."),
    ("easy","Explain code","d = {k:v for k,v in zip(keys, vals)}","Creates a dictionary by zipping two lists. Equivalent to dict(zip(keys, vals))."),
    ("medium","Rename and explain","def x(a):\n    b = []\n    for c in a:\n        if c not in b:\n            b.append(c)\n    return b","def deduplicate(items):\n    seen = []\n    for item in items:\n        if item not in seen:\n            seen.append(item)\n    return seen","Removes duplicates while preserving order."),
    ("medium","Rename variables","def calc(t, r, n):\n    return t * (1 + r/n)**(n)","def compound_interest(principal, rate, periods):\n    return principal * (1 + rate/periods)**(periods)","Compound interest formula."),
    ("medium","Simplify control flow","if x > 0:\n    if x < 10:\n        return True\n    else:\n        return False\nelse:\n    return False","return 0 < x < 10","Nested ifs collapse to a single range check."),
    ("medium","Rename and explain","def proc(d):\n    r = {}\n    for k, v in d.items():\n        r[v] = k\n    return r","def invert_dict(d):\n    inverted = {}\n    for key, value in d.items():\n        inverted[value] = key\n    return inverted","Swaps keys and values in a dictionary."),
    ("medium","Explain suspicious behavior","import os; os.system('rm -rf /tmp/*')","Deletes all files in /tmp. This is a cleanup command but could be destructive if run with elevated privileges."),
    ("medium","Rename variables","res = list(map(lambda x: x.strip().lower(), data))","cleaned = [item.strip().lower() for item in data]","Strips whitespace and lowercases each string. List comp is cleaner than map+lambda."),
    ("medium","Simplify nested ternary","x = 'yes' if a > 0 else ('no' if a < 0 else 'maybe')","if a > 0:\n    result = 'yes'\nelif a < 0:\n    result = 'no'\nelse:\n    result = 'maybe'","Classifies positive/negative/zero."),
    ("hard","Deobfuscate full function","def _0x3f(a):\n    _0x1 = []\n    _0x2 = 0\n    while _0x2 < len(a):\n        _0x3 = a[_0x2]\n        _0x4 = 1\n        while _0x2 + _0x4 < len(a) and a[_0x2 + _0x4] == _0x3:\n            _0x4 += 1\n        _0x1.append((_0x3, _0x4))\n        _0x2 += _0x4\n    return _0x1","def run_length_encode(data):\n    encoded = []\n    i = 0\n    while i < len(data):\n        char = data[i]\n        count = 1\n        while i + count < len(data) and data[i + count] == char:\n            count += 1\n        encoded.append((char, count))\n        i += count\n    return encoded","Run-length encoding. Identifies consecutive identical characters and counts them."),
    ("hard","Identify the algorithm","def mystery(arr):\n    for i in range(1, len(arr)):\n        key = arr[i]\n        j = i - 1\n        while j >= 0 and arr[j] > key:\n            arr[j + 1] = arr[j]\n            j -= 1\n        arr[j + 1] = key\n    return arr","Insertion sort. Shifts elements right until the correct position for each item is found. O(n^2) average."),
    ("medium","Explain code pattern","with open('f.txt') as f:\n    lines = [l.strip() for l in f if l.strip()]","Reads file, strips whitespace, removes empty lines. Context manager ensures file is closed."),
    ("medium","Rename variables","def h(d, k, v=None):\n    return d.get(k, v)","def safe_get(dictionary, key, default=None):\n    return dictionary.get(key, default)","Safely retrieves a dict value with a default."),
    ("medium","Explain obfuscated regex","import re\np = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')","Email validation regex. Matches: alphanumeric/special chars + @ + domain + dot + 2+ letter TLD. Not RFC-complete but practical."),
    ("hard","Deobfuscate data pipeline","result = reduce(lambda a,b: {**a,**b}, [dict(zip(range(i*3,(i+1)*3),v)) for i,v in enumerate(chunks)])","Flattens chunked data into a single numbered dict. Each chunk of 3 items gets sequential keys. Merges all dicts using reduce."),
    ("medium","Simplify verbose code","output = []\nfor x in data:\n    if x is not None:\n        if x > 0:\n            output.append(x * 2)","output = [x * 2 for x in data if x is not None and x > 0]","Filters positive non-null values and doubles them."),
    ("medium","Explain callback pattern","def on_complete(f):\n    def wrapper(*a, **k):\n        result = f(*a, **k)\n        log(f'{f.__name__} done: {result}')\n        return result\n    return wrapper","Decorator that logs function completion and return value."),
    ("hard","Decode bit manipulation","def count_set_bits(n):\n    c = 0\n    while n:\n        n &= n - 1\n        c += 1\n    return c","Brian Kernighan's algorithm. Each n &= n-1 clears the lowest set bit. Counts 1-bits in binary representation."),
    ("medium","Explain metaclass usage","class Singleton(type):\n    _instances = {}\n    def __call__(cls, *args, **kwargs):\n        if cls not in cls._instances:\n            cls._instances[cls] = super().__call__(*args, **kwargs)\n        return cls._instances[cls]","Metaclass that ensures only one instance of a class exists. Uses class registry to track instances."),
    ("medium","Rename variables","def f(s, n=2):\n    return [s[i:i+n] for i in range(0, len(s), n)]","def chunk_string(text, size=2):\n    return [text[i:i+size] for i in range(0, len(text), size)]","Splits string into fixed-size chunks."),
    ("easy","Explain one-liner","result = any(x > 10 for x in numbers)","Returns True if any element in numbers is greater than 10. Short-circuits on first match."),
    ("easy","Explain code","data = sorted(data, key=lambda x: x['name'])","Sorts list of dicts by the 'name' field alphabetically."),
    ("medium","Identify vulnerability","import pickle\ndata = pickle.loads(user_input)","Deserializing untrusted input with pickle is dangerous — pickle can execute arbitrary code. Use json or a safe serializer."),
    ("medium","Explain decorator","@property\ndef full_name(self):\n    return f'{self.first} {self.last}'","Makes full_name accessible as an attribute (obj.full_name) instead of a method call (obj.full_name())."),
    ("hard","Decode async pattern","async def fetch_all(urls):\n    async with aiohttp.ClientSession() as session:\n        tasks = [fetch(session, url) for url in urls]\n        return await asyncio.gather(*tasks)","Concurrent HTTP fetching. Creates one session, spawns parallel tasks for each URL, gathers all results."),
    ("medium","Explain generator","def fib():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b","Infinite Fibonacci generator. Uses yield to produce values lazily. Call next() or iterate to get values."),
    ("medium","Simplify code","try:\n    val = d['key']\nexcept KeyError:\n    val = default","val = d.get('key', default)","dict.get() is equivalent and more readable than try/except for KeyError."),
    ("hard","Deobfuscate data transform","data = {k: list(v) for k, v in groupby(sorted(items, key=keyfunc), key=keyfunc)}","Groups items by a key function. Sorts first (required by groupby), then groups consecutive items with same key into lists."),
    ("medium","Explain bitwise ops","flags = READ | WRITE\nif flags & WRITE:\n    print('writable')","Bitwise OR to set flags, bitwise AND to check flags. Each permission is a single bit position."),
]
for diff,intent,ctx,answer,*rest in deobf:
    add("deobfuscation",diff,intent,ctx,answer)

# === FACTUAL (33) ===
factual = [
    ("easy","Capital of France?","","Paris."),
    ("easy","Speed of light?","","Approximately 299,792,458 meters per second (or ~300,000 km/s)."),
    ("easy","Largest planet in solar system?","","Jupiter."),
    ("easy","Who wrote 1984?","","George Orwell."),
    ("easy","Chemical formula for water?","","H2O."),
    ("easy","How many continents?","","Seven: Africa, Antarctica, Asia, Australia, Europe, North America, South America."),
    ("easy","Currency of Japan?","","Japanese Yen (JPY)."),
    ("easy","Tallest mountain?","","Mount Everest, 8,849 meters above sea level."),
    ("medium","What is the Turing test?","","A test where a human evaluator converses with both a human and machine. If the evaluator can't reliably distinguish them, the machine passes."),
    ("medium","What is CRISPR?","","A gene-editing tool that uses a guide RNA to direct the Cas9 enzyme to cut DNA at specific locations, enabling precise genetic modifications."),
    ("medium","What causes tides?","","Gravitational pull of the Moon and Sun on Earth's oceans. The Moon has the dominant effect due to proximity."),
    ("medium","What is the Heisenberg uncertainty principle?","","You cannot simultaneously know both the exact position and momentum of a particle. Measuring one more precisely makes the other less certain."),
    ("medium","What year did the Berlin Wall fall?","","1989."),
    ("medium","What is compound interest?","","Interest calculated on both the initial principal and previously accumulated interest. Formula: A = P(1 + r/n)^(nt)."),
    ("medium","What is the placebo effect?","","A real improvement in health from a treatment with no active ingredient, caused by the patient's belief in the treatment."),
    ("medium","What does GDP measure?","","Gross Domestic Product: the total monetary value of all goods and services produced within a country in a specific time period."),
    ("medium","What is the James Webb Space Telescope?","","An infrared space telescope launched Dec 2021, orbiting the Sun at the L2 point. It studies the earliest galaxies, star formation, and exoplanet atmospheres."),
    ("medium","What is the difference between weather and climate?","","Weather is short-term atmospheric conditions (days). Climate is the long-term average weather pattern of a region (30+ years)."),
    ("hard","What is Gödel's incompleteness theorem?","","Any consistent formal system powerful enough to express arithmetic contains true statements that cannot be proven within the system."),
    ("hard","What is the CAP theorem?","","A distributed system can guarantee at most 2 of 3: Consistency (all nodes see same data), Availability (every request gets a response), Partition tolerance (works despite network splits)."),
    ("hard","What is quantum entanglement?","","When two particles become correlated such that measuring one instantly determines the state of the other, regardless of distance. No information is transmitted faster than light."),
    ("hard","What is the P vs NP problem?","","Can every problem whose solution can be quickly verified (NP) also be quickly solved (P)? Most computer scientists believe P ≠ NP but no proof exists."),
    ("hard","What was Project MKUltra?","","A CIA program (1953-1973) that conducted illegal human experiments with LSD and other methods for mind control research."),
    ("easy","What is the capital of Australia?","","Canberra (not Sydney or Melbourne)."),
    ("easy","How many bones in adult human body?","","206."),
    ("easy","What does DNA stand for?","","Deoxyribonucleic acid."),
    ("medium","What is the butterfly effect?","","The idea that small changes in initial conditions can lead to large differences in outcomes, from chaos theory. Named by Edward Lorenz."),
    ("medium","What is the prisoner's dilemma?","","Two rational individuals might not cooperate even when it's in their mutual interest, because each fears the other will defect."),
    ("medium","Who painted the Mona Lisa?","","Leonardo da Vinci, circa 1503-1519."),
    ("medium","What is Schrödinger's cat?","","A thought experiment: a cat in a sealed box is simultaneously alive and dead until observed, illustrating quantum superposition's absurdity at macro scale."),
    ("hard","What is homomorphic encryption?","","Encryption that allows computations on encrypted data without decrypting it first. The result, when decrypted, matches the result of operations on the plaintext."),
    ("hard","What is the halting problem?","","It's impossible to write a general program that determines whether any arbitrary program will eventually halt or run forever. Proven by Turing in 1936."),
    ("hard","What happened in the Cambrian explosion?","","A rapid diversification of complex animal life ~540 million years ago. Most major animal phyla appeared in fossils over ~20 million years. Causes debated: oxygen rise, new genes (Hox), ecological arms race."),
]
for diff,intent,ctx,answer in factual:
    add("factual",diff,intent,ctx,answer)

# === CONCISION (33) ===
concision = [
    ("easy","Explain git in one sentence","","Git is a distributed version control system that tracks changes in files, allowing multiple people to work on the same codebase simultaneously."),
    ("easy","What is Docker in under 20 words","","Docker packages applications with their dependencies into portable containers that run consistently on any system."),
    ("easy","What is a REST API in one sentence","","An interface where clients send HTTP requests to endpoints to create, read, update, or delete resources, typically using JSON."),
    ("easy","Explain SQL in 15 words","","A language for querying and manipulating structured data in relational databases using tables, rows, and columns."),
    ("easy","What is Linux in under 20 words","","An open-source Unix-like operating system kernel that powers most servers, Android phones, and supercomputers."),
    ("easy","What does RAM do in one sentence","","RAM provides fast temporary storage for data your computer is actively using; it's cleared when powered off."),
    ("easy","Explain cookies (web) in 15 words","","Small text files websites store in your browser to remember preferences, sessions, and tracking data."),
    ("easy","What is a variable in programming in one sentence","","A named container that stores a value in memory so you can reference and modify it later."),
    ("medium","Explain Kubernetes without jargon","","Kubernetes automatically manages containerised applications — starting them, scaling them up or down, and restarting them if they crash."),
    ("medium","What is serverless in 20 words","","You write code, the cloud runs it on demand. No servers to manage, you pay per execution."),
    ("medium","Explain OAuth to a 10-year-old","","Instead of giving a app your password, you show a special pass from Google that says 'this app can see X'."),
    ("medium","What is a load balancer in one sentence","","A traffic controller that distributes incoming requests across multiple servers so no single server gets overwhelmed."),
    ("medium","Explain microservices vs monolith concisely","","Monolith: one big app. Microservices: many small apps talking over a network. Monolith is simpler to start; microservices scale better."),
    ("medium","What is CI/CD in under 25 words","","Automated pipeline: code changes are built, tested, and deployed without manual steps. CI = integrate often. CD = deploy often."),
    ("medium","Explain technical debt in one sentence","","Shortcuts in code that speed up delivery now but create maintenance headaches later — like a loan with interest."),
    ("medium","What is a hash function in 20 words","","A function that converts any input into a fixed-size string. Same input always gives same output. One-way: can't reverse it."),
    ("hard","Explain distributed systems challenge in one sentence","","Getting multiple independent computers to agree on shared state while any of them can fail or lose network connectivity."),
    ("hard","What is eventual consistency","","A guarantee that if no new updates are made, all replicas will eventually converge to the same value — but not immediately."),
    ("hard","Explain the CAP theorem without jargon","","In a distributed system, when the network splits, you must choose: either everyone sees the same data (but some can't respond) or everyone responds (but might see stale data)."),
    ("hard","What is a data lake vs data warehouse in 20 words","","Lake: raw dumps of any data, structured or not. Warehouse: cleaned, structured data optimised for SQL queries."),
    ("medium","Explain Agile in one sentence","","Iterative approach: build in small increments, get feedback frequently, adapt plans based on what you learn."),
    ("medium","What is a container (vs VM) concisely","","Container: shares host OS kernel, lightweight, starts in seconds. VM: full guest OS, heavier, better isolation."),
    ("medium","Explain Git rebase vs merge in 20 words","","Merge: creates a merge commit combining branches. Rebase: replays your commits on top of the target branch, creating linear history."),
    ("medium","What is a CDN in one sentence","","A network of servers worldwide that caches your content closer to users, reducing latency and load on your origin server."),
    ("medium","Explain idempotency in 15 words","","Calling an operation multiple times produces the same result as calling it once. Safe to retry."),
    ("hard","What is a vector database in 20 words","","A database optimised for storing and searching high-dimensional vectors (embeddings) using similarity metrics like cosine distance."),
    ("hard","Explain the actor model in one sentence","","A concurrency model where actors are independent units that communicate only by sending messages, never sharing state directly."),
    ("hard","What is a service mesh in 25 words","","Infrastructure layer that handles service-to-service communication: load balancing, encryption, retries, observability — without changing application code."),
    ("medium","Explain JSON in 15 words","","A lightweight text format for structured data using key-value pairs and arrays. Human-readable, language-independent."),
    ("medium","What is an API gateway in 20 words","","A single entry point that routes API requests to appropriate backend services, handling auth, rate limiting, and logging."),
    ("medium","Explain NoSQL in one sentence","","Databases that don't use traditional table structure — they store data as documents, key-value pairs, graphs, or wide columns."),
    ("hard","What is CRDT in 20 words","","Conflict-free Replicated Data Types: data structures that can be updated independently on multiple nodes and always converge without coordination."),
    ("hard","Explain zero-knowledge proof in one sentence","","Proving you know something without revealing what you know — like proving you're over 18 without showing your birthdate."),
]
for diff,intent,ctx,answer in concision:
    add("concision",diff,intent,ctx,answer)

# === MULTI-TURN (33) ===
mt_data = [
    ("easy","Debug a function",[
        {"user":"My function returns None instead of the result. def add(a,b): a+b","assistant":"You're missing the return statement. Change it to: def add(a, b): return a + b"},
        {"user":"That fixed it, thanks! Now how do I add type hints?","assistant":"def add(a: int, b: int) -> int: return a + b. The type hints are after the parameter names with a colon."}
    ]),
    ("easy","Set up Python project",[
        {"user":"How do I set up a new Python project?","assistant":"Create a directory, run python -m venv .venv, activate it, then pip install your dependencies. Add a requirements.txt with pip freeze > requirements.txt."},
        {"user":"What about a .gitignore?","assistant":"Add .venv/, __pycache__/, *.pyc, .env, dist/, *.egg-info/. GitHub has a Python template at github.com/github/gitignore."}
    ]),
    ("medium","Explain recursion",[
        {"user":"What is recursion?","assistant":"A function that calls itself. Every recursive function needs a base case (when to stop) and a recursive case (the self-call)."},
        {"user":"Give me an example","assistant":"Factorial: def fact(n): return 1 if n <= 1 else n * fact(n-1). The base case is n<=1 returns 1. The recursive case multiplies n by fact(n-1)."},
        {"user":"What happens without a base case?","assistant":"Infinite recursion. Python raises RecursionError after hitting the default limit (1000 calls). You can change it with sys.setrecursionlimit() but usually you should fix the logic instead."}
    ]),
    ("medium","Learn SQL",[
        {"user":"I want to learn SQL. Where should I start?","assistant":"Start with SELECT, FROM, WHERE. Practice on SQLite (no setup needed). Try sqlbolt.com for interactive exercises. After basics, learn JOINs and GROUP BY."},
        {"user":"What's the most important thing to understand?","assistant":"Thinking in sets, not loops. SQL operates on entire tables at once. Instead of 'for each row, check X', write 'WHERE X'. This mental shift is the biggest hurdle from imperative programming."},
        {"user":"Any common mistakes?","assistant":"Forgetting that NULL != NULL (use IS NULL instead). Not using GROUP BY with aggregates. SELECT * in production code. Not indexing columns you frequently filter on."}
    ]),
    ("medium","Fix slow database query",[
        {"user":"My SQL query takes 30 seconds. SELECT * FROM orders WHERE customer_id = 42 AND status = 'pending'","assistant":"Add a composite index: CREATE INDEX idx_orders_cust_status ON orders(customer_id, status). This lets the database find matching rows without scanning the whole table."},
        {"user":"How do I know if it's working?","assistant":"Run EXPLAIN SELECT * FROM orders WHERE customer_id = 42 AND status = 'pending'. If you see 'Index Scan' instead of 'Seq Scan', the index is being used. Execution time should drop dramatically."},
        {"user":"What if I also filter by date?","assistant":"Extend the index: CREATE INDEX idx_orders_cust_status_date ON orders(customer_id, status, order_date). Order matters — put the most selective columns first. But don't over-index; each index slows writes."}
    ]),
    ("hard","Debug memory leak",[
        {"user":"My Python app memory grows over time. How do I find the leak?","assistant":"Start with tracemalloc: import tracemalloc; tracemalloc.start(). After some work, call tracemalloc.get_traced_memory() to see current/peak. For detailed analysis, take snapshots with tracemalloc.take_snapshot() and compare."},
        {"user":"I found it's a growing dict. How do I fix it?","assistant":"If it's a cache, use functools.lru_cache(maxsize=N) or cachetools.TTLCache. If it's a global accumulator, check if you're forgetting to remove items. For event handlers, ensure you're not holding references to destroyed objects."},
        {"user":"What about circular references?","assistant":"Python's garbage collector handles most circular references, but not all (especially with __del__ methods). Use weakref.ref() for back-references. gc.get_referrers(obj) shows what's keeping an object alive."}
    ]),
    ("easy","Convert JSON to CSV",[
        {"user":"How do I convert a JSON file to CSV?","assistant":"With Python: import json, csv. Load the JSON with json.load(), then write with csv.DictWriter. If the JSON is nested, flatten it first with pandas.json_normalize() or manual flattening."},
        {"user":"What if the JSON has nested objects?","assistant":"Use pandas: import pandas as pd; df = pd.read_json('file.json'); df.to_csv('file.csv', index=False). For complex nesting, pd.json_normalize(data, sep='_') flattens nested keys with underscore separators."}
    ]),
    ("medium","Set up Docker for development",[
        {"user":"I want to containerize my Python Flask app. Where do I start?","assistant":"Create a Dockerfile: FROM python:3.12-slim, WORKDIR /app, COPY requirements.txt ., RUN pip install -r requirements.txt, COPY . ., EXPOSE 5000, CMD [\"python\", \"app.py\"]. Then: docker build -t myapp . && docker run -p 5000:5000 myapp."},
        {"user":"How do I handle environment variables?","assistant":"Pass with docker run -e DATABASE_URL=... or use a .env file with docker run --env-file .env. Never hardcode secrets in the Dockerfile. For development, mount the code as a volume: -v $(pwd):/app."},
        {"user":"What about docker-compose?","assistant":"Define services in docker-compose.yml. Good for multi-container setups (app + database + redis). Run with docker-compose up. Use 'volumes' for code mounting in dev and 'depends_on' for startup order."}
    ]),
    ("medium","Learn regex",[
        {"user":"I need to learn regex for parsing log files. Basics?","assistant":"Key patterns: . (any char), * (0+ of previous), + (1+), ? (optional), \\d (digit), \\w (word char), \\s (whitespace), [abc] (character class), (group). Use re.findall(pattern, text) for extraction."},
        {"user":"How do I match an IP address?","assistant":"Pattern: r'\\b\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\b'. The \\b is a word boundary. For stricter matching (0-255 only), it gets complex — use ipaddress module instead."},
        {"user":"What about extracting timestamps?","assistant":"r'\\d{4}-\\d{2}-\\d{2}[T ]\\d{2}:\\d{2}:\\d{2}' matches ISO-like timestamps. Add groups with () to extract parts: r'(\\d{4})-(\\d{2})-(\\d{2})' gives year, month, day separately via re.findall() or re.match().groups()."}
    ]),
    ("hard","Implement authentication",[
        {"user":"I need to add user authentication to my API. What's the simplest secure approach?","assistant":"Use JWT tokens. On login: verify credentials, generate a JWT with user ID and expiry (use PyJWT library). On each request: validate the JWT in middleware. Store passwords hashed with bcrypt (passlib library)."},
        {"user":"How do I handle token refresh?","assistant":"Issue two tokens: access_token (short-lived, 15min) and refresh_token (long-lived, 7 days). Store refresh_token in httpOnly cookie. When access_token expires, client sends refresh_token to get a new pair. Rotate refresh tokens on each use."},
        {"user":"What about rate limiting?","assistant":"Use Flask-Limiter or implement token bucket. Limit by IP + user ID. Example: 100 requests/minute for authenticated, 20/minute for login endpoint. Return 429 Too Many Requests with Retry-After header."}
    ]),
    ("easy","Understand Python lists vs tuples",[
        {"user":"What's the difference between a list and a tuple in Python?","assistant":"Lists are mutable (can change), tuples are immutable (can't change). Lists use [], tuples use (). Tuples are slightly faster and can be dict keys; lists can't."},
        {"user":"When should I use each?","assistant":"Use tuples for fixed collections (coordinates, RGB colors, function returns). Use lists when the collection changes (items to process, results to accumulate). If in doubt, start with a tuple and switch to list if you need mutability."}
    ]),
    ("medium","Deploy to production",[
        {"user":"I have a Flask app ready to deploy. What's the simplest production setup?","assistant":"Use Gunicorn as the WSGI server behind Nginx. Gunicorn handles Python processes, Nginx handles static files and SSL. For easy hosting: Railway, Fly.io, or a VPS with systemd."},
        {"user":"How do I handle SSL?","assistant":"Get a free certificate from Let's Encrypt (certbot). Nginx terminates SSL and proxies to Gunicorn on localhost. Auto-renew with certbot renew --dry-run. Or use Cloudflare as a reverse proxy (free tier includes SSL)."},
        {"user":"What about database migrations?","assistant":"Use Alembic (for SQLAlchemy) or Django's built-in migrations. Always test migrations on a copy of production data first. Run alembic upgrade head on deploy, alembic downgrade -1 to rollback. Never delete migration files."}
    ]),
    ("medium","Write unit tests",[
        {"user":"How should I structure unit tests for a Python project?","assistant":"Put tests in a tests/ directory mirroring your source structure. Name files test_*.py. Use pytest (simpler than unittest). Each test function tests one behavior. Group related tests in classes."},
        {"user":"What should I actually test?","assistant":"Test edge cases (empty input, None, boundary values), error conditions (invalid input raises exceptions), and the main happy path. Don't test implementation details — test behavior from the outside. Aim for confidence, not 100% coverage."},
        {"user":"How do I test code that depends on external services?","assistant":"Mock external calls with unittest.mock.patch or pytest-mock. For database: use an in-memory SQLite or testcontainers. For APIs: use responses library to mock HTTP. Don't test the external service itself — test how your code handles its responses."}
    ]),
    ("easy","Explain version control",[
        {"user":"Why should I use version control?","assistant":"It tracks every change to your code, lets you undo mistakes, collaborate without conflicts, and maintain a complete history. Git is the standard. Every professional developer uses it."},
        {"user":"What's a commit?","assistant":"A snapshot of your changes with a message describing what you did. Think of it as a save point in a game — you can always go back to it. Write clear commit messages: 'Add user login' not 'fix stuff'."}
    ]),
    ("medium","Handle errors properly",[
        {"user":"How should I handle errors in Python?","assistant":"Use try/except for expected failures (file not found, network timeout). Let unexpected errors propagate. Catch specific exceptions, not bare except. Always log the error."},
        {"user":"When should I raise vs return None?","assistant":"Raise when the caller must handle it (authentication failure, invalid input). Return None/default when the absence of a result is normal (search found nothing). Be consistent within your codebase."},
        {"user":"What about custom exceptions?","assistant":"Create them for domain-specific errors: class UserNotFoundError(Exception): pass. Inherit from Exception, not BaseException. Add useful attributes: raise UserNotFoundError(user_id=42). Keep the hierarchy shallow."}
    ]),
    ("hard","Optimize Python performance",[
        {"user":"My Python script is too slow. How do I find the bottleneck?","assistant":"Profile first: python -m cProfile -o profile.prof script.py, then visualize with snakeviz profile.prof. Don't guess — measure. 90% of the time, the bottleneck is in 10% of the code."},
        {"user":"The bottleneck is a loop processing millions of items","assistant":"Replace the loop with numpy operations (vectorisation), or use list comprehensions instead of append loops. For truly compute-heavy work, use multiprocessing (not threading — Python's GIL limits CPU-bound threading)."},
        {"user":"What if I need it even faster?","assistant":"Options: rewrite the hot path in C (via ctypes or Cython), use PyPy instead of CPython, or use Numba JIT (@numba.jit decorator). For I/O-bound work, use asyncio or concurrent.futures.ThreadPoolExecutor."}
    ]),
    ("easy","Understand HTTP status codes",[
        {"user":"What do HTTP status codes mean?","assistant":"2xx = success (200 OK, 201 Created). 3xx = redirect (301 permanent, 302 temporary). 4xx = client error (400 bad request, 401 unauthorized, 403 forbidden, 404 not found). 5xx = server error (500 internal, 502 bad gateway, 503 unavailable)."},
        {"user":"Which ones should my API return?","assistant":"POST success: 201 Created. GET success: 200. Delete: 204 No Content. Bad input: 400. Not authenticated: 401. Not authorised: 403. Not found: 404. Rate limited: 429. Server crash: 500."}
    ]),
    ("medium","Choose a database",[
        {"user":"Should I use PostgreSQL or MongoDB for my new project?","assistant":"PostgreSQL unless you have a specific reason not to. It handles structured data, JSON (jsonb), full-text search, and scales well. MongoDB only if your data is genuinely document-shaped with no relations."},
        {"user":"What about SQLite?","assistant":"SQLite is perfect for development, small apps (< 100k writes/day), and embedded use. It's a single file — no server needed. Don't use it for multi-user concurrent writes. Start with SQLite, migrate to Postgres when you need it."},
        {"user":"When would Redis make sense?","assistant":"Redis is a cache, not a primary database. Use it for: session storage, rate limiting counters, pub/sub messaging, and caching expensive query results. Always have a persistent database behind it."}
    ]),
    ("medium","Code review best practices",[
        {"user":"How should I do code reviews?","assistant":"Focus on: correctness (does it work?), clarity (can I understand it in 6 months?), edge cases (what could break?), and security (any injection/leak risks?). Don't nitpick style — use a linter for that."},
        {"user":"How big should a review be?","assistant":"Under 400 lines changed. Anything bigger should be split into stacked PRs. Review within 24 hours. If you can't understand it from the PR description + code, the PR needs better documentation."}
    ]),
    ("easy","Understand APIs",[
        {"user":"What is a REST API?","assistant":"A way for programs to talk over HTTP. You send requests (GET, POST, PUT, DELETE) to URLs (endpoints) and get back data (usually JSON). Like a waiter: you (client) order from a menu (API docs), the kitchen (server) prepares it."},
        {"user":"How do I test one?","assistant":"Use curl for quick tests: curl https://api.example.com/users. For more complex testing, use httpie (http GET https://api.example.com/users) or Postman for GUI-based testing with saved requests."}
    ]),
]
for diff,intent,turns in mt_data:
    add_mt(diff,intent,turns)

# Write output
os.makedirs("data/canonical", exist_ok=True)
random.shuffle(examples)  # shuffle for diversity
# Re-assign IDs after shuffle to maintain ordering
for i, ex in enumerate(examples):
    ex["id"] = f"can_{i+1:03d}"

with open("data/canonical/phase9_pilot_300.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Generated {len(examples)} canonical examples")
# Domain counts
from collections import Counter
domains = Counter(ex["domain"] for ex in examples)
for d, c in sorted(domains.items()):
    print(f"  {d}: {c}")
