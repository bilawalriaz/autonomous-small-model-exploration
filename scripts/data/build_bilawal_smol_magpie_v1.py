#!/usr/bin/env python3
"""Generate bilawal_smol_magpie_v1 canonical dataset.
Small-model-native SFT mixture: 1000 examples, 6 categories.
Usage: python scripts/data/build_bilawal_smol_magpie_v1.py
"""
import json, random, os, sys

random.seed(42)

# Target distribution (1000 total)
# 37.5% general concise = 375
# 20% coding = 200
# 15% structured extraction = 150
# 10% reasoning = 100
# 12.5% domain-specific (gamefaq + deobfuscation) = 125
# 5% anti-slop = 50

examples = []
seq = 0

def add(domain, diff, intent, ctx, answer, constraints=None):
    global seq
    seq += 1
    examples.append({
        "id": f"bsmagpie_{seq:04d}",
        "domain": domain,
        "source": "synthetic",
        "difficulty": diff,
        "user_intent": intent,
        "context": ctx,
        "ideal_answer": answer,
        "constraints": constraints or ["concise"],
        "metadata": {"origin_dataset": "bilawal_smol_magpie_v1", "license": "MIT", "notes": ""}
    })

def add_mt(diff, intent, turns, constraints=None):
    global seq
    seq += 1
    examples.append({
        "id": f"bsmagpie_{seq:04d}",
        "domain": "multi_turn",
        "source": "synthetic",
        "difficulty": diff,
        "user_intent": intent,
        "context": "",
        "ideal_answer": turns[-1]["assistant"],
        "constraints": constraints or [],
        "metadata": {"origin_dataset": "bilawal_smol_magpie_v1", "license": "MIT", "notes": ""},
        "turns": turns
    })

# ── GENERAL CONCISE (375) ──────────────────────────────────────
# Mix of Q&A, practical help, explanations, task completion
general_qa = [
    ("easy", "What's the boiling point of water?", "", "100°C at sea level (212°F)."),
    ("easy", "How many days in a leap year?", "", "366 days."),
    ("easy", "What does HTTP stand for?", "", "HyperText Transfer Protocol."),
    ("easy", "What is a byte?", "", "8 bits of data."),
    ("easy", "What's the currency of Japan?", "", "Japanese Yen (JPY)."),
    ("easy", "Convert 5 miles to km.", "", "5 × 1.609 = 8.045 km."),
    ("easy", "What's 15% of 200?", "", "30."),
    ("easy", "What's the capital of Canada?", "", "Ottawa (not Toronto or Vancouver)."),
    ("easy", "How many sides does a hexagon have?", "", "6."),
    ("easy", "What year did WW2 end?", "", "1945."),
    ("medium", "Explain DNS in one sentence.", "", "DNS translates domain names into IP addresses so browsers can find servers."),
    ("medium", "What is a REST API?", "", "An interface where clients send HTTP requests to endpoints to create, read, update, or delete resources, typically using JSON."),
    ("medium", "What is containerisation?", "", "Packaging an application with its dependencies into a lightweight, portable container that runs consistently on any system."),
    ("medium", "Explain the difference between TCP and UDP.", "", "TCP: reliable, ordered, connection-based (web, email). UDP: fast, no guarantees, connectionless (streaming, gaming)."),
    ("medium", "What is a load balancer?", "", "A traffic controller that distributes incoming requests across multiple servers to prevent overload."),
    ("medium", "Explain OAuth to a non-technical person.", "", "Instead of sharing your password with an app, you show a special pass from Google that says 'this app can access X'."),
    ("medium", "What is serverless computing?", "", "Running code on-demand without managing servers. You pay per execution, not per server."),
    ("medium", "What's the difference between SQL and NoSQL?", "", "SQL: fixed schemas, tables, relationships. NoSQL: flexible schemas (documents, key-value, graphs)."),
    ("medium", "Explain what a CDN does.", "", "Caches content on servers worldwide, delivering it from the closest location to reduce latency."),
    ("medium", "What is infrastructure as code?", "", "Managing infrastructure through version-controlled config files instead of manual setup."),
    ("medium", "What is a microservice?", "", "A small, independently deployable service handling one business capability, communicating via APIs."),
    ("medium", "Explain the CAP theorem simply.", "", "A distributed system can guarantee at most 2 of 3: Consistency, Availability, Partition tolerance."),
    ("medium", "What is a VPN?", "", "Encrypts your internet connection and routes it through a remote server, hiding your real IP."),
    ("medium", "What does CI/CD mean?", "", "Continuous Integration/Continuous Deployment: automatically build, test, and deploy code changes."),
    ("medium", "Explain technical debt.", "", "Shortcuts in code that speed up delivery now but create maintenance headaches later."),
    ("medium", "What is a hash function?", "", "Converts any input to a fixed-size string. Same input always gives same output. One-way."),
    ("medium", "Explain event-driven architecture.", "", "Components communicate by producing and consuming events rather than direct calls."),
    ("medium", "What is a service mesh?", "", "Infrastructure layer handling service-to-service communication: load balancing, encryption, retries."),
    ("medium", "What is an API gateway?", "", "Single entry point routing API requests to backend services, handling auth and rate limiting."),
    ("medium", "Explain what a data lake is.", "", "A repository storing raw data in its native format (structured or unstructured) until needed."),
    ("medium", "What is idempotency?", "", "Calling an operation multiple times produces the same result as calling it once. Safe to retry."),
    ("medium", "What is a vector database?", "", "A database optimised for storing and searching high-dimensional vectors using similarity metrics."),
    ("hard", "Explain zero-knowledge proofs.", "", "Proving you know something without revealing what you know — like proving you're over 18 without showing your birthdate."),
    ("hard", "What is CRDT?", "", "Conflict-free Replicated Data Types: data structures that converge without coordination across nodes."),
    ("hard", "Explain the actor model.", "", "Independent actors communicate only by sending messages, never sharing state directly."),
    ("hard", "What is the halting problem?", "", "Impossible to write a general program that determines whether any arbitrary program will halt. Proven by Turing in 1936."),
    ("hard", "What is homomorphic encryption?", "", "Computing on encrypted data without decrypting it. The result, when decrypted, matches plaintext operations."),
]
practical = [
    ("easy", "How to store leftover pasta?", "", "Airtight container in fridge, 3-5 days. Toss with a little oil to prevent sticking."),
    ("easy", "How to remove a coffee stain?", "", "Blot (don't rub). Apply vinegar + dish soap 1:1. Wait 15 min. Wash cold."),
    ("easy", "Best way to peel garlic?", "", "Crush with flat side of knife. Skin slides off."),
    ("easy", "How to unclog a drain naturally?", "", "1/2 cup baking soda, then 1/2 cup vinegar. Cover 30 min. Flush with boiling water."),
    ("easy", "How to soft-boil an egg?", "", "6 minutes in boiling water, then ice bath immediately."),
    ("medium", "How to set up a home network securely?", "", "Change default password. Use WPA3. Update firmware. Guest network. Disable WPS. Enable firewall."),
    ("medium", "How to prepare for a tech interview?", "", "Research company + stack. Review job description. Practice coding. Prepare questions. Test video setup."),
    ("medium", "How to make pour-over coffee?", "", "15g coffee per 250ml, medium-fine grind. Water 93-96°C. Bloom 30ml/30s. Pour in circles. 3-4 min total."),
    ("medium", "How to start investing?", "", "Open stocks ISA (UK, 20k/yr tax-free). Low-cost global index fund. Invest regularly. Keep 3-6 months cash first."),
    ("medium", "How to sleep better?", "", "Same bedtime/wake time. No screens 1hr before. Cool dark room. No caffeine after 2pm. Exercise earlier."),
    ("medium", "How to reduce food waste?", "", "Plan meals. Make shopping list. Store properly. Use leftovers. Freeze before expiry. Compost scraps."),
    ("medium", "How to set up automated backups?", "", "Linux: rsync + cron. Mac: Time Machine. Windows: File History. Test restoring monthly. Keep one offsite."),
    ("medium", "What to do if identity stolen?", "", "Freeze bank accounts. Report Action Fraud. Check credit report. Change passwords. Contact HMRC. Crime reference number."),
    ("medium", "How to choose a password manager?", "", "Bitwarden (free, open source) or 1Password (polished). Enable 2FA. Generate unique passwords per site."),
    ("medium", "How to set up a VPN?", "", "Choose Mullvad or ProtonVPN. Install app. Connect to nearest server. Enable kill switch. Test at ipleak.net."),
    ("medium", "Explain DNS to non-technical.", "", "The internet's phone book. When you type google.com, DNS translates it to an IP address so your browser connects."),
    ("medium", "How to fix slow computer?", "", "Check Task Manager for high-CPU apps. Clear temp files. Check startup programs. Run disk cleanup. Check malware. Consider SSD."),
    ("medium", "How to prepare for power outage?", "", "Charge devices. Fill water containers. Stock canned food + torch + batteries. Fill car. Cash in hand."),
    ("medium", "Ergonomic home office setup?", "", "Monitor at eye level, arm's length. Elbows 90°. Feet flat. Lumbar support. Break every hour."),
    ("medium", "How to cook rice perfectly?", "", "Rinse until water clear. 1:1.5 rice:water. Bring to boil, simmer covered 18 min. Rest 5 min. Don't peek."),
    ("medium", "Explain HTTPS to non-technical.", "", "Browser and server do a secret handshake. Server proves identity via certificate. All data scrambled with secret code. Padlock icon means it worked."),
    ("medium", "What should I check buying used laptop?", "", "Battery health, screen pixels, keyboard, ports, SSD health, RAM, warranty. Test under load 15 min."),
    ("medium", "How to deploy a Python web app?", "", "Gunicorn as WSGI server behind Nginx. Free SSL from Let's Encrypt. Use systemd for process management."),
    ("medium", "What to look for renting vs buying?", "", "Renting: flexible, no maintenance, no equity. Buying: builds equity, stable payments, ties to location."),
    ("medium", "How to reduce energy bills?", "", "LED bulbs. Smart thermostat. Draft excluders. Insulate loft. Shorter showers. Turn off standby devices."),
]
task_completion = [
    ("easy", "Write a haiku about rain.", "", "Droplets on glass panes\nPuddles mirror grey above\nUmbrellas blossom"),
    ("easy", "Write a limerick about a cat.", "", "A cat who resided in Bray\nSleeps twenty hours every day\nWhen asked to go walk\nShe lets out a squawk\nThen rolls over, purring away"),
    ("easy", "Write a thank you message for a gift.", "", "Thank you so much for the lovely gift — it was incredibly thoughtful and I really appreciate it."),
    ("easy", "Write a professional out-of-office reply.", "", "Thanks for your email. I'm currently out of the office with limited access to email. I'll respond on my return. For urgent matters, please contact [colleague]."),
    ("medium", "Write a cover letter intro for a DevOps role.", "", "I'm a platform engineer with 6+ years experience in Azure, Kubernetes, and Terraform, with a track record of reducing deployment times by 80% and infrastructure costs by 30%."),
    ("medium", "Write a project README intro.", "", "Brief description, installation steps, usage examples, and contribution guidelines — the four things every README needs upfront."),
    ("medium", "Write a concise product description for wireless earbuds.", "", "Wireless earbuds with active noise cancellation, 8-hour battery (32 with case), IPX4 water resistance, and Bluetooth 5.3. Weighs 5g per bud."),
    ("medium", "Write a complaint email about a late delivery.", "", "Order #12345 placed on Jan 5 was promised by Jan 10. It's now Jan 18 with no update. Please advise on expected delivery or process a refund."),
    ("medium", "Write a 3-sentence bio for a tech blog.", "", "Platform engineer focused on infrastructure automation and AI tooling. Previously at Dotdigital and Capita. Writes about DevOps, small models, and building reliable systems."),
    ("medium", "Write a meeting agenda for a sprint retrospective.", "", "1. What went well (10 min). 2. What didn't go well (10 min). 3. Action items (10 min). 4. Vote on top priority (5 min)."),
    ("medium", "Write a commit message for fixing a login bug.", "", "Fix login timeout on slow connections by increasing auth request timeout from 5s to 15s and adding retry logic."),
    ("medium", "Write a PR description for adding dark mode.", "", "Adds system-preference-aware dark mode using CSS custom properties. Covers all components including charts and code blocks. Tested on Chrome, Firefox, Safari."),
    ("medium", "Write a 50-word description of this project.", "", "MI-Atlas maps the internal control surfaces of small language models through causal interventions — ablation, steering, patching, and LoRA training. Each claim carries a metric, counterfactual, and confidence level."),
    ("medium", "Write a Slack message telling the team about a deployment.", "", "Deployed v2.3.1 to prod. Changes: auth timeout fix, dark mode, new API rate limiting. Monitoring closely — ping me if anything looks off."),
    ("medium", "Write a polite decline for a meeting invite.", "", "Thanks for the invite. I can't make this slot — could you share the notes or a recording? Happy to provide input async."),
    ("medium", "Write a 3-step tutorial for setting up Python venv.", "", "1. python -m venv .venv\n2. source .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows)\n3. pip install -r requirements.txt"),
    ("medium", "Write an incident report summary.", "", "At 14:32 UTC, the API gateway began returning 503 errors due to database connection exhaustion. Root cause: missing connection pool limit. Resolution: added pool cap of 10 connections. Time to resolution: 23 minutes."),
    ("medium", "Write a concise error message for invalid email.", "", "Please enter a valid email address (e.g., user@domain.com)."),
    ("medium", "Write a documentation snippet for a REST endpoint.", "", "GET /api/users/:id — Returns a single user object. Requires authentication. Returns 404 if user not found. Response: {id, name, email, role, created_at}."),
    ("medium", "Write a README installation section.", "", "## Installation\n\n```bash\ngit clone https://github.com/bilawalriaz/project.git\ncd project\npython -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt\n```"),
    ("medium", "Write a concise LinkedIn summary.", "", "Platform engineer specialising in infrastructure automation, Kubernetes, and AI tooling. Building reliable systems at scale. Previously at Dotdigital, Capita, and Yourgene Health."),
    ("medium", "Write a weekly status update.", "", "This week: deployed auth fix, completed dark mode feature, started rate limiting work. Blockers: none. Next week: finish rate limiting, begin API v2 planning."),
    ("medium", "Write a docstring for a function that parses CSV.", "", '"""Parse CSV text into a list of dictionaries.\n\nArgs:\n    text: Raw CSV string with header row.\n    delimiter: Column separator (default: comma).\n\nReturns:\n    List of dicts keyed by header names.\n"""'),
]

# Add general examples with variety
for diff, intent, ctx, answer in general_qa[:37]:  # 37 general Q&A
    add("general", diff, intent, ctx, answer)
for diff, intent, ctx, answer in practical[:22]:  # 22 practical
    add("general", diff, intent, ctx, answer)
for diff, intent, ctx, answer in task_completion[:20]:  # 20 task completion
    add("general", diff, intent, ctx, answer)

# Multi-turn general conversations
general_mt = [
    ("easy", "What's Python?", [
        {"user": "What is Python?", "assistant": "A high-level programming language known for readability and versatility. Used for web development, data science, automation, and AI."},
        {"user": "Should I learn it as my first language?", "assistant": "Yes. Python's syntax is close to English, the community is huge, and it's useful in almost every field. Start with the official tutorial at python.org."}
    ]),
    ("medium", "How to learn git", [
        {"user": "I want to learn git. Where do I start?", "assistant": "Start with: git init, git add, git commit, git push. Practice on a throwaway repo. Try learngitbranching.js.org for interactive exercises."},
        {"user": "What about branches?", "assistant": "A branch is a pointer to a commit. Create with git branch feature or git checkout -b feature. Merge with git merge. Don't be afraid of branches — they're cheap."},
        {"user": "When should I use rebase vs merge?", "assistant": "Merge: creates a merge commit, preserves history. Rebase: replays commits for linear history. Use rebase for local cleanup, merge for shared branches."}
    ]),
    ("medium", "Learn SQL basics", [
        {"user": "I need to learn SQL for work. Basics?", "assistant": "Start with SELECT, FROM, WHERE. Practice on SQLite (no setup). sqlbolt.com for interactive exercises. After basics, learn JOINs and GROUP BY."},
        {"user": "What's the most important concept?", "assistant": "Thinking in sets, not loops. SQL operates on entire tables at once. Instead of 'for each row, check X', write 'WHERE X'."},
        {"user": "Common mistakes?", "assistant": "NULL != NULL (use IS NULL). Forgetting GROUP BY with aggregates. SELECT * in production. Not indexing columns you filter on frequently."}
    ]),
    ("medium", "Understand async programming", [
        {"user": "What is async programming?", "assistant": "Instead of waiting for slow operations (network, disk), async lets your program do other work while waiting. Like a chef starting multiple dishes instead of cooking one at a time."},
        {"user": "When should I use it vs threading?", "assistant": "Async for I/O-bound tasks (web requests, file reads). Threading for similar but with blocking libraries. Multiprocessing for CPU-bound work. In Python, asyncio is cooperative concurrency."},
        {"user": "Show me a simple example.", "assistant": "import asyncio\n\nasync def fetch(url):\n    # simulate network call\n    await asyncio.sleep(1)\n    return f'Result from {url}'\n\nasync def main():\n    results = await asyncio.gather(\n        fetch('a.com'), fetch('b.com'), fetch('c.com')\n    )\n    print(results)  # runs all 3 concurrently, takes ~1s\n\nasyncio.run(main())"}
    ]),
    ("medium", "Learn regex", [
        {"user": "I need to learn regex for log parsing. Basics?", "assistant": "Key patterns: . (any char), * (0+), + (1+), \\d (digit), \\w (word char), [abc] (class), (group). Use re.findall(pattern, text) for extraction."},
        {"user": "How to match an IP address?", "assistant": "r'\\b\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\b'. For strict 0-255 validation, it gets complex — use the ipaddress module instead."},
        {"user": "How to extract timestamps?", "assistant": "r'\\d{4}-\\d{2}-\\d{2}[T ]\\d{2}:\\d{2}:\\d{2}' matches ISO-like timestamps. Use () groups to extract parts: re.match(r'(\\d{4})-(\\d{2})-(\\d{2})', text).groups()."}
    ]),
    ("medium", "Choose a database", [
        {"user": "PostgreSQL or MongoDB for my app?", "assistant": "PostgreSQL unless you have a specific reason not to. Handles structured data, JSON, full-text search. MongoDB only if data is genuinely document-shaped with no relations."},
        {"user": "What about SQLite?", "assistant": "Perfect for development, small apps (<100k writes/day), embedded use. Single file, no server. Start with SQLite, migrate to Postgres when you need it."},
        {"user": "When would Redis make sense?", "assistant": "Redis is a cache, not primary database. Use for: session storage, rate limiting, pub/sub, caching expensive query results. Always have persistent DB behind it."}
    ]),
    ("medium", "Set up monitoring", [
        {"user": "How to monitor my Python web app?", "assistant": "Start with structured logging (structlog). Add health check endpoints. Prometheus + Grafana for metrics. Sentry for error tracking."},
        {"user": "What metrics to track?", "assistant": "Four golden signals: latency (request duration), traffic (requests/sec), errors (error rate %), saturation (CPU/memory). Also: DB query time, queue depth."},
        {"user": "How to set up alerts?", "assistant": "Alert on symptoms, not causes. Good: error rate >1% for 5min, p99 latency >2s, CPU >80% for 10min. Bad: every individual error. Use PagerDuty or Grafana alerting."}
    ]),
    ("medium", "Write unit tests", [
        {"user": "How to structure unit tests in Python?", "assistant": "Put tests in tests/ mirroring source. Name files test_*.py. Use pytest. Each test function tests one behavior. Group related tests in classes."},
        {"user": "What should I test?", "assistant": "Edge cases (empty, None, boundaries), error conditions (invalid input raises), happy path. Don't test implementation details — test behavior from outside."},
        {"user": "How to test external service dependencies?", "assistant": "Mock with unittest.mock.patch or pytest-mock. For DB: in-memory SQLite or testcontainers. For APIs: responses library. Don't test the external service itself."}
    ]),
    ("medium", "Handle errors in Python", [
        {"user": "How should I handle errors?", "assistant": "try/except for expected failures (FileNotFoundError, TimeoutError). Let unexpected errors propagate. Catch specific exceptions, not bare except. Always log."},
        {"user": "When to raise vs return None?", "assistant": "Raise when caller must handle it (auth failure, invalid input). Return None when absence is normal (search found nothing). Be consistent."},
        {"user": "Custom exceptions?", "assistant": "class UserNotFoundError(Exception): pass. Inherit from Exception, not BaseException. Add useful attributes: raise UserNotFoundError(user_id=42). Keep hierarchy shallow."}
    ]),
    ("medium", "Set up Docker for development", [
        {"user": "Containerize my Flask app?", "assistant": "Dockerfile: FROM python:3.12-slim, WORKDIR /app, COPY requirements.txt ., RUN pip install -r requirements.txt, COPY . ., EXPOSE 5000, CMD [\"python\", \"app.py\"]. docker build -t myapp . && docker run -p 5000:5000 myapp."},
        {"user": "How to handle env vars?", "assistant": "Pass with docker run -e DATABASE_URL=... or --env-file .env. Never hardcode secrets in Dockerfile. For dev, mount code as volume: -v $(pwd):/app."},
        {"user": "What about docker-compose?", "assistant": "Define services in docker-compose.yml. Good for multi-container (app + DB + redis). docker-compose up. Use volumes for code mounting and depends_on for startup order."}
    ]),
    ("medium", "Learn about caching", [
        {"user": "When should I use caching?", "assistant": "Cache when: same data read frequently, data doesn't change often, source is slow (DB, API). Redis for app data, CDN for static assets, browser cache for client-side."},
        {"user": "How to invalidate the cache?", "assistant": "TTL (auto-expire after N seconds — simplest), write-through (update on every write), event-driven (publish changes, subscribers invalidate). Prefer TTL unless you need real-time."},
        {"user": "What about cache stampede?", "assistant": "When cache expires and 100 requests try to rebuild simultaneously. Fix: lock/mutex (one rebuilds, others wait), probabilistic early expiration, or stale-while-revalidate."}
    ]),
    ("medium", "Code review best practices", [
        {"user": "How to do code reviews?", "assistant": "Focus on: correctness, clarity, edge cases, security. Don't nitpick style — use a linter. Review within 24 hours."},
        {"user": "How big should a review be?", "assistant": "Under 400 lines changed. Bigger should be split into stacked PRs. If you can't understand it from the PR description + code, it needs better docs."}
    ]),
    ("easy", "Understand HTTP status codes", [
        {"user": "What do HTTP status codes mean?", "assistant": "2xx success (200 OK, 201 Created). 3xx redirect (301 permanent, 302 temporary). 4xx client error (400 bad request, 401 unauthorized, 403 forbidden, 404 not found). 5xx server error (500, 502, 503)."},
        {"user": "Which should my API return?", "assistant": "POST success: 201. GET success: 200. Delete: 204 No Content. Bad input: 400. Not authenticated: 401. Not authorised: 403. Not found: 404. Rate limited: 429. Server crash: 500."}
    ]),
    ("medium", "Concurrency vs parallelism", [
        {"user": "Difference between concurrency and parallelism?", "assistant": "Concurrency: dealing with multiple tasks (interleaving). Parallelism: executing multiple tasks simultaneously. One chef juggling pots = concurrent. Two chefs = parallel."},
        {"user": "How does this apply to Python?", "assistant": "Python has GIL: only one thread runs bytecode at a time. Threading gives concurrency (I/O-bound). Multiprocessing gives parallelism (CPU-bound). asyncio is cooperative concurrency."}
    ]),
    ("easy", "Understand environment variables", [
        {"user": "What are environment variables?", "assistant": "Key-value pairs set outside code, accessible via os.environ. Use for config that changes between environments (DB URLs, API keys). Never hardcode secrets."},
        {"user": "How to use them in Python?", "assistant": "Read: os.environ.get('DB_URL', 'default'). Set: os.environ['DB_URL'] = 'value'. For .env files: python-dotenv: from dotenv import load_dotenv; load_dotenv()."}
    ]),
    ("medium", "Frontend framework choice", [
        {"user": "React vs Vue vs Svelte?", "assistant": "React: largest ecosystem, most jobs, JSX. Vue: gentler learning, template syntax, great docs. Svelte: compile-time, least boilerplate, best small-app performance. For jobs: React. For personal: Svelte or Vue."},
        {"user": "I want to build something quickly.", "assistant": "Svelte with SvelteKit. Less code, built-in routing and SSR, good defaults. If you already know React, Next.js is fastest."}
    ]),
    ("medium", "Debug slow Python script", [
        {"user": "My Python script is slow. How to find the bottleneck?", "assistant": "Profile: python -m cProfile -o prof.py script.py, then snakeviz prof.py. Don't guess — measure. 90% of the time, the bottleneck is in 10% of code."},
        {"user": "The bottleneck is a loop over millions of items.", "assistant": "Replace with numpy (vectorisation) or list comprehensions instead of append loops. For CPU-bound: multiprocessing. For I/O-bound: asyncio or concurrent.futures.ThreadPoolExecutor."},
        {"user": "Need it even faster?", "assistant": "Rewrite hot path in C (ctypes or Cython), use PyPy, or Numba JIT (@numba.jit). For I/O: asyncio or ThreadPoolExecutor."}
    ]),
]
for diff, intent, turns in general_mt:
    add_mt(diff, intent, turns)

# Fill remaining general with more Q&A
extra_general = [
    ("easy", "What is Linux?", "", "An open-source Unix-like OS kernel powering most servers, Android phones, and supercomputers."),
    ("easy", "What does RAM do?", "", "Fast temporary storage for data your computer actively uses. Cleared when powered off."),
    ("easy", "What is a cookie (web)?", "", "Small text file websites store in your browser to remember preferences and sessions."),
    ("easy", "What is a variable?", "", "A named container storing a value in memory so you can reference and modify it later."),
    ("easy", "What is an API?", "", "A way for programs to talk to each other. Like a waiter between you (client) and the kitchen (server)."),
    ("medium", "Explain Kubernetes simply.", "", "Automatically manages containers — starting, scaling, and restarting them across a cluster."),
    ("medium", "What is a CDN?", "", "Network of servers worldwide caching content closer to users for faster delivery."),
    ("medium", "Explain OAuth simply.", "", "Instead of giving apps your password, you show a pass from Google saying 'this app can see X'."),
    ("medium", "What is a proxy server?", "", "Sits between you and the internet, forwarding requests. Can hide IP, cache content, or block access."),
    ("medium", "What is NoSQL?", "", "Databases that don't use tables — store data as documents, key-value pairs, graphs, or columns."),
    ("medium", "Explain distributed systems challenge.", "", "Getting multiple independent computers to agree on shared state while any can fail or lose connectivity."),
    ("medium", "What is eventual consistency?", "", "If no new updates, all replicas eventually converge to the same value — but not immediately."),
    ("medium", "What is a data warehouse?", "", "Cleaned, structured data optimised for SQL queries and analytics. Opposite of a data lake."),
    ("hard", "Explain Gödel's incompleteness theorem.", "", "Any consistent formal system powerful enough for arithmetic contains true statements that cannot be proven within it."),
    ("hard", "What is the P vs NP problem?", "", "Can every problem whose solution is quickly verified (NP) also be quickly solved (P)? Most believe no, but no proof exists."),
    ("hard", "What is quantum entanglement?", "", "Two particles correlated such that measuring one instantly determines the other's state, regardless of distance."),
    ("medium", "What is a hash map?", "", "Data structure mapping keys to values using a hash function for O(1) average lookup. Python dict is a hash map."),
    ("medium", "What is Big O notation?", "", "Describes how an algorithm's time/space scales with input size. O(1) constant, O(n) linear, O(n²) quadratic, O(log n) logarithmic."),
    ("medium", "Explain the observer pattern.", "", "An object (subject) maintains a list of dependents (observers) and notifies them of state changes automatically."),
    ("medium", "What is a singleton?", "", "A design pattern ensuring only one instance of a class exists. Usually implemented with a class variable storing the instance."),
    ("medium", "Explain dependency injection.", "", "Instead of creating dependencies inside a class, they're passed in from outside. Makes testing and swapping implementations easier."),
    ("medium", "What is middleware?", "", "Software sitting between request and response, processing each request (auth, logging, rate limiting) before it reaches the handler."),
]
for diff, intent, ctx, answer in extra_general:
    add("general", diff, intent, ctx, answer)

# ── CODING (200) ──────────────────────────────────────
coding_funcs = [
    ("easy", "Write is_even(n)", "def is_even(n): return n % 2 == 0"),
    ("easy", "Reverse a string", "def reverse_str(s): return s[::-1]"),
    ("easy", "Sum a list", "def sum_list(nums): return sum(nums)"),
    ("easy", "Find max value", "def find_max(nums):\n    m = nums[0]\n    for n in nums[1:]:\n        if n > m: m = n\n    return m"),
    ("easy", "Celsius to Fahrenheit", "def c_to_f(c): return c * 9/5 + 32"),
    ("easy", "Palindrome check", "def is_palindrome(s):\n    t = s.replace(' ','').lower()\n    return t == t[::-1]"),
    ("easy", "Greeting function", "def greet(name): return f'Hello, {name}!'"),
    ("easy", "List to CSV", "def to_csv(items): return ', '.join(str(x) for x in items)"),
    ("easy", "Check prime", "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True"),
    ("easy", "Factorial", "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)"),
    ("medium", "Flatten nested list", "def flatten(lst):\n    r = []\n    for x in lst:\n        if isinstance(x, list): r.extend(flatten(x))\n        else: r.append(x)\n    return r"),
    ("medium", "Binary search", "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        m = (lo+hi)//2\n        if arr[m] == target: return m\n        elif arr[m] < target: lo = m+1\n        else: hi = m-1\n    return -1"),
    ("medium", "Merge sorted lists", "def merge_sorted(a, b):\n    r = []; i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]: r.append(a[i]); i += 1\n        else: r.append(b[j]); j += 1\n    r.extend(a[i:]); r.extend(b[j:])\n    return r"),
    ("medium", "Word frequency", "from collections import Counter\ndef word_freq(text): return dict(Counter(text.lower().split()))"),
    ("medium", "Validate parentheses", "def is_valid_parens(s):\n    st = []; m = {')':'(',']':'[','}':'{'}\n    for c in s:\n        if c in m:\n            if not st or st[-1] != m[c]: return False\n            st.pop()\n        elif c in m.values(): st.append(c)\n    return len(st) == 0"),
    ("medium", "Fibonacci memoized", "from functools import lru_cache\n@lru_cache(None)\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)"),
    ("medium", "Group anagrams", "from collections import defaultdict\ndef group_anagrams(words):\n    g = defaultdict(list)\n    for w in words: g[''.join(sorted(w))].append(w)\n    return list(g.values())"),
    ("medium", "Password validator", "import re\ndef is_valid_password(pw):\n    return len(pw)>=8 and bool(re.search(r'[A-Z]',pw)) and bool(re.search(r'[a-z]',pw)) and bool(re.search(r'\\d',pw))"),
    ("medium", "Chunk list", "def chunk_list(lst, n): return [lst[i:i+n] for i in range(0, len(lst), n)]"),
    ("medium", "Caesar cipher", "def caesar(text, shift):\n    r = []\n    for c in text:\n        if c.isalpha():\n            base = ord('A') if c.isupper() else ord('a')\n            r.append(chr((ord(c)-base+shift)%26+base))\n        else: r.append(c)\n    return ''.join(r)"),
    ("medium", "LRU Cache", "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, cap):\n        self.c = OrderedDict(); self.cap = cap\n    def get(self, k):\n        if k not in self.c: return -1\n        self.c.move_to_end(k); return self.c[k]\n    def put(self, k, v):\n        if k in self.c: self.c.move_to_end(k)\n        self.c[k] = v\n        if len(self.c) > self.cap: self.c.popitem(last=False)"),
    ("medium", "Stack class", "class Stack:\n    def __init__(self): self.items = []\n    def push(self, x): self.items.append(x)\n    def pop(self): return self.items.pop()\n    def peek(self): return self.items[-1]\n    def is_empty(self): return len(self.items) == 0"),
    ("medium", "Queue from stacks", "class Queue:\n    def __init__(self): self.inn=[]; self.out=[]\n    def enqueue(self, x): self.inn.append(x)\n    def dequeue(self):\n        if not self.out:\n            while self.inn: self.out.append(self.inn.pop())\n        return self.out.pop()"),
    ("medium", "Retry decorator", "import functools, time\ndef retry(n=3, d=1):\n    def deco(f):\n        @functools.wraps(f)\n        def w(*a, **k):\n            for i in range(n):\n                try: return f(*a, **k)\n                except:\n                    if i==n-1: raise\n                    time.sleep(d)\n        return w\n    return deco"),
    ("medium", "Parse CSV to dicts", "def parse_csv(text):\n    lines = text.strip().split('\\n')\n    h = [x.strip() for x in lines[0].split(',')]\n    return [dict(zip(h,[v.strip() for v in l.split(',')])) for l in lines[1:]]"),
    ("medium", "Flatten dict with dots", "def flatten_dict(d, prefix=''):\n    r = {}\n    for k, v in d.items():\n        key = f'{prefix}.{k}' if prefix else k\n        if isinstance(v, dict): r.update(flatten_dict(v, key))\n        else: r[key] = v\n    return r"),
    ("medium", "Debounce function", "import threading\ndef debounce(fn, delay):\n    timer = None\n    def wrapper(*a, **k):\n        nonlocal timer\n        if timer: timer.cancel()\n        timer = threading.Timer(delay/1000, lambda: fn(*a, **k))\n        timer.start()\n    return wrapper"),
    ("medium", "Timer context manager", "import time\nfrom contextlib import contextmanager\n@contextmanager\ndef timer(label='block'):\n    start = time.perf_counter()\n    yield\n    print(f'{label}: {time.perf_counter()-start:.4f}s')"),
    ("hard", "Trie data structure", "class Trie:\n    def __init__(self): self.ch={}; self.end=False\n    def insert(self, w):\n        n=self\n        for c in w:\n            if c not in n.ch: n.ch[c]=Trie()\n            n=n.ch[c]\n        n.end=True\n    def search(self, w):\n        n=self._find(w)\n        return n and n.end\n    def starts_with(self, p): return self._find(p) is not None\n    def _find(self, p):\n        n=self\n        for c in p:\n            if c not in n.ch: return None\n            n=n.ch[c]\n        return n"),
    ("hard", "Token bucket limiter", "import time\nclass TokenBucket:\n    def __init__(s,rate,cap): s.rate=rate; s.cap=cap; s.tokens=cap; s.t=time.monotonic()\n    def consume(s,n=1):\n        now=time.monotonic(); s.tokens=min(s.cap,s.tokens+(now-s.t)*s.rate); s.t=now\n        if s.tokens>=n: s.tokens-=n; return True\n        return False"),
    ("hard", "Expression parser", "import re, operator as op\ndef eval_expr(s):\n    toks=re.findall(r'\\d+|[+*-]',s); nums=[int(toks[0])]; ops=[]; i=1\n    while i<len(toks):\n        if toks[i]=='*': nums[-1]*=int(toks[i+1])\n        else: nums.append(int(toks[i+1])); ops.append(toks[i])\n        i+=2\n    r=nums[0]\n    for j,o in enumerate(ops):\n        r=(op.add if o=='+' else op.sub)(r,nums[j+1])\n    return r"),
    ("hard", "Quicksort in-place", "def qs(a,lo=0,hi=None):\n    if hi is None: hi=len(a)-1\n    if lo<hi:\n        p=partition(a,lo,hi); qs(a,lo,p-1); qs(a,p+1,hi)\n    return a\ndef partition(a,lo,hi):\n    pv=a[hi]; i=lo\n    for j in range(lo,hi):\n        if a[j]<pv: a[i],a[j]=a[j],a[i]; i+=1\n    a[i],a[hi]=a[hi],a[i]\n    return i"),
]
for diff, intent, code in coding_funcs:
    add("coding", diff, intent, intent, code)

# Coding Q&A and debugging
coding_qa = [
    ("easy", "Fix missing return", "Bug: def double(n): n * 2", "def double(n): return n * 2"),
    ("easy", "Fix off-by-one", "Bug: for i in range(len(lst)-1): print(lst[i])", "for i in range(len(lst)): print(lst[i])"),
    ("medium", "Fix race condition", "Bug: global count += 1 not thread-safe", "import threading\nlock = threading.Lock()\ndef increment():\n    global count\n    with lock: count += 1"),
    ("medium", "Fix mutable default", "Bug: def add(x, lst=[]): lst.append(x)", "def add(x, lst=None):\n    if lst is None: lst = []\n    lst.append(x)\n    return lst"),
    ("medium", "Fix infinite loop", "Bug: count=10; while count>0: print(count)", "count = 10\nwhile count > 0:\n    print(count)\n    count -= 1"),
    ("medium", "Fix memory leak", "Bug: cache = {} grows unboundedly", "from functools import lru_cache\n@lru_cache(maxsize=1000)\ndef process(key): return heavy_transform(key)"),
    ("medium", "Explain *args and **kwargs", "", "*args collects positional args into a tuple. **kwargs collects keyword args into a dict. Allows variable-argument functions."),
    ("medium", "Explain is vs ==", "", "'==' checks value equality. 'is' checks identity (same object). Use 'is' for None: x is None."),
    ("medium", "Explain generators", "", "Functions that yield values lazily. Use yield instead of return. Call next() or iterate. Memory-efficient for large sequences."),
    ("medium", "Explain decorators", "", "Functions that wrap other functions. @decorator is sugar for func = decorator(func). Common: logging, timing, caching, auth."),
    ("medium", "Explain list vs tuple", "", "Lists: mutable, []. Tuples: immutable, (). Use tuples for fixed data (coordinates), lists for changing data (items to process)."),
    ("medium", "What is a context manager?", "", "Manages resource lifecycle via with statement. __enter__ on entry, __exit__ on exit. Ensures cleanup even on errors."),
    ("medium", "Explain metaclasses", "", "Classes that create classes. type is the default metaclass. Use for singleton pattern, class registration, validation."),
    ("medium", "Fix SQL injection", "Bug: f\"SELECT * FROM users WHERE name = '{input}'\"", "cursor.execute('SELECT * FROM users WHERE name = ?', (input,))"),
    ("hard", "Implement producer-consumer", "", "import threading, collections\nclass BoundedBuffer:\n    def __init__(s, cap): s.q=collections.deque(); s.cap=cap; s.lock=threading.Lock(); s.not_full=threading.Condition(s.lock); s.not_empty=threading.Condition(s.lock)\n    def put(s, item):\n        with s.not_full:\n            while len(s.q)>=s.cap: s.not_full.wait()\n            s.q.append(item); s.not_empty.notify()\n    def get(s):\n        with s.not_empty:\n            while not s.q: s.not_empty.wait()\n            item=s.q.popleft(); s.not_full.notify()\n            return item"),
    ("hard", "Implement merge sort", "def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(a, b):\n    r = []; i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]: r.append(a[i]); i += 1\n        else: r.append(b[j]); j += 1\n    r.extend(a[i:]); r.extend(b[j:])\n    return r"),
]
for item in coding_qa:
    if len(item) == 4:
        diff, intent, ctx, answer = item
    else:
        diff, intent, answer = item
        ctx = intent
    add("coding", diff, intent, ctx, answer)

# Multi-turn coding
coding_mt = [
    ("easy", "Debug function returning None", [
        {"user": "My function returns None. def add(a,b): a+b", "assistant": "Missing return. Change to: def add(a, b): return a + b"},
        {"user": "Fixed! Now add type hints.", "assistant": "def add(a: int, b: int) -> int: return a + b"},
    ]),
    ("medium", "Learn list comprehensions", [
        {"user": "What is a list comprehension?", "assistant": "Compact list creation: [expr for item in iterable if cond]. Example: [x**2 for x in range(10) if x%2==0] gives [0, 4, 16, 36, 64]."},
        {"user": "Can I use if-else?", "assistant": "Yes, but syntax differs: [x if x > 0 else 0 for x in numbers]. if-else goes before 'for'; filter-only if goes after."},
    ]),
    ("medium", "Implement authentication", [
        {"user": "Simple secure auth for my API?", "assistant": "JWT tokens. On login: verify credentials, generate JWT with user ID and expiry (PyJWT). On each request: validate JWT in middleware. Hash passwords with bcrypt."},
        {"user": "How to handle refresh?", "assistant": "Two tokens: access (15min) and refresh (7 days). Refresh in httpOnly cookie. When access expires, send refresh to get new pair. Rotate refresh tokens on use."},
        {"user": "Rate limiting?", "assistant": "Flask-Limiter or token bucket. Limit by IP + user ID. 100/min authenticated, 20/min login. Return 429 with Retry-After header."},
    ]),
    ("medium", "Deploy to production", [
        {"user": "Simplest production setup for Flask?", "assistant": "Gunicorn behind Nginx. Gunicorn handles Python, Nginx handles static files + SSL. Free SSL from Let's Encrypt (certbot)."},
        {"user": "How to handle SSL?", "assistant": "certbot for Let's Encrypt. Nginx terminates SSL, proxies to Gunicorn on localhost. Auto-renew with certbot renew --dry-run."},
        {"user": "Database migrations?", "assistant": "Alembic for SQLAlchemy. Test migrations on production data copy first. alembic upgrade head to deploy, downgrade -1 to rollback."},
    ]),
]
for diff, intent, turns in coding_mt:
    add_mt(diff, intent, turns)

# ── STRUCTURED EXTRACTION / JSON (150) ──────────────────────────────
json_extracts = [
    ("easy", "Extract person info", "John Smith, 34, Manchester, software engineer at TechCorp.", '{"name":"John Smith","age":34,"city":"Manchester","occupation":"software engineer","employer":"TechCorp"}'),
    ("easy", "Parse product listing", "Product: Laptop, Price: 899.99 USD, Stock: 42", '{"product":"Laptop","price":{"amount":899.99,"currency":"USD"},"stock":42}'),
    ("easy", "Extract event details", "Conference March 15 2026, ICC Birmingham, Registration 8:30am, Keynote 9am, 150 GBP", '{"event":"conference","date":"2026-03-15","venue":"ICC Birmingham","price":{"amount":150,"currency":"GBP"}}'),
    ("easy", "Extract date and time", "Meeting: Tuesday 14 Jan 2026 at 2:30 PM GMT", '{"day":"Tuesday","date":"2026-01-14","time":"14:30","timezone":"GMT"}'),
    ("easy", "Extract color RGB", "Red: rgb(255,0,0), Green: rgb(0,128,0), Blue: rgb(0,0,255)", '{"colors":[{"name":"Red","r":255,"g":0,"b":0},{"name":"Green","r":0,"g":128,"b":0},{"name":"Blue","r":0,"g":0,"b":255}]}'),
    ("medium", "Extract restaurant", "Golden Dragon, Chinese, 42 High St. Mon-Sat 11:30-22:00, Sun 12-21. 18 GBP avg. Vegetarian. Phone 0161 234 5678.", '{"name":"Golden Dragon","cuisine":"Chinese","avg_meal":{"amount":18,"currency":"GBP"},"vegetarian":true,"phone":"0161 234 5678"}'),
    ("medium", "Parse messy address", "Flat 3b, 147 St. Georges Rd, Salford, Greater Manchester, M3 6DN, UK", '{"apartment":"3b","number":147,"street":"St. Georges Rd","city":"Salford","postcode":"M3 6DN","country":"UK"}'),
    ("medium", "Extract weather", "Manchester: 14C high, 8C low, 72% humidity, 12mph NW wind, UV 3, overcast", '{"city":"Manchester","temp":{"high":14,"low":8},"humidity":72,"wind":{"mph":12,"dir":"NW"},"uv":3}'),
    ("medium", "Extract movie", "Inception (2010), dir Nolan, DiCaprio/Gordon-Levitt/Page, 148min, IMDB 8.8", '{"title":"Inception","year":2010,"director":"Christopher Nolan","runtime_min":148,"imdb":8.8}'),
    ("medium", "Extract email metadata", "From alice@ex.com to bob@ex.com, cc carol@ex.com. Subject: Q4 Report. Date: 5 Jan 2026 09:15. Priority: High.", '{"from":"alice@example.com","to":["bob@example.com"],"cc":["carol@example.com"],"subject":"Q4 Report","priority":"High"}'),
    ("medium", "Extract sports stats", "Liverpool 2-1 Arsenal. Goals: Salah 23', Nunez 67', Saka 45'. Possession 58-42%", '{"home":"Liverpool","away":"Arsenal","score":"2-1","possession":{"Liverpool":58,"Arsenal":42}}'),
    ("medium", "Extract flight", "BA2490 LHR->DXB. Dep 2026-04-10 21:35. Arr 2026-04-11 07:45. 6h10m. Boeing 777.", '{"flight":"BA2490","from":"LHR","to":"DXB","depart":"2026-04-10T21:35","arrive":"2026-04-11T07:45","duration":"6h10m"}'),
    ("medium", "Extract invoice", "INV-2026-042, 1 Mar. Acme Corp. Web Dev 40hrs@75, Hosting@120, Domain@15. Subtotal 3135, VAT 627, Total 3762 GBP", '{"invoice":"INV-2026-042","client":"Acme Corp","subtotal":3135,"vat":627,"total":3762}'),
    ("medium", "Extract git log", "a3b2c1d 'Fix login' by Alice 2026-01-10. d4e5f6a 'Add dark mode' by Bob 2026-01-12.", '{"commits":[{"hash":"a3b2c1d","message":"Fix login","author":"Alice","date":"2026-01-10"},{"hash":"d4e5f6a","message":"Add dark mode","author":"Bob","date":"2026-01-12"}]}'),
    ("medium", "Extract test results", "42 passed, 3 failed, 1 skipped. Failed: test_login (line 87), test_rate_limit, test_export. Coverage: 87.3%", '{"passed":42,"failed":3,"skipped":1,"coverage":87.3}'),
    ("medium", "Extract pagination", "Page 2 of 10, per_page 20, total 198. Links: first /api/items?page=1, next /api/items?page=3, last /api/items?page=10", '{"page":2,"pages":10,"per_page":20,"total":198}'),
    ("medium", "Extract CSS", "body { font-family: sans-serif; margin: 0; padding: 20px; background: #f5f5f5 }", '{"body":{"font-family":"sans-serif","margin":"0","padding":"20px","background":"#f5f5f5"}}'),
    ("medium", "Extract cron schedule", "Backup daily 2am. Report weekly Mon 9am. Cleanup 1st of month midnight. Health check every 5min.", '{"jobs":[{"name":"backup","schedule":"0 2 * * *"},{"name":"report","schedule":"0 9 * * 1"},{"name":"cleanup","schedule":"0 0 1 * *"},{"name":"health_check","schedule":"*/5 * * * *"}]}'),
    ("medium", "Extract meeting notes", "Standup 2026-01-15. Alice: finished auth, blocked on API key. Bob: deploying staging. Carol: 80% test coverage.", '{"meeting":"Standup","date":"2026-01-15","updates":[{"person":"Alice","done":"auth","blocked":"API key"},{"person":"Bob","plan":"deploy staging"},{"person":"Carol","progress":"80% coverage"}]}'),
    ("medium", "Extract error log", "2026-01-15 14:32:01 ERROR [api-gateway] Connection refused db-primary:5432 after 3 retries. /app/db.py:45", '{"timestamp":"2026-01-15T14:32:01","level":"ERROR","service":"api-gateway","error":"Connection refused","target":"db-primary:5432"}'),
    ("medium", "Extract Docker compose", "services: web (nginx:1.25, port 80:80), api (build ./api, port 3000:3000), postgres (postgres:16, vol pgdata)", '{"services":{"web":{"image":"nginx:1.25","ports":["80:80"]},"api":{"build":"./api","ports":["3000:3000"]},"postgres":{"image":"postgres:16","volumes":["pgdata"]}}}'),
    ("medium", "Extract package.json", "name my-app, version 2.1.0, deps express ^4.18, react ^18.2, devDeps jest ^29.7", '{"name":"my-app","version":"2.1.0","dependencies":{"express":"^4.18","react":"^18.2"},"devDependencies":{"jest":"^29.7"}}'),
    ("medium", "Convert table to JSON", "ID|Name|Score\n1|Alice|95\n2|Bob|87\n3|Carol|92", '[{"id":1,"name":"Alice","score":95},{"id":2,"name":"Bob","score":87},{"id":3,"name":"Carol","score":92}]'),
    ("medium", "Extract SQL schema", "CREATE TABLE users (id SERIAL PRIMARY KEY, username VARCHAR(50) UNIQUE NOT NULL, email VARCHAR(100) NOT NULL, active BOOLEAN DEFAULT true)", '{"table":"users","columns":[{"name":"id","type":"SERIAL"},{"name":"username","type":"VARCHAR(50)","constraints":["UNIQUE","NOT NULL"]},{"name":"email","type":"VARCHAR(100)","constraint":"NOT NULL"},{"name":"active","type":"BOOLEAN","default":true}]}'),
    ("medium", "Extract complex schema", "User types: Admin(read,write,delete), Editor(read,write), Viewer(read). Fields: id(UUID), email(unique), name, created_at(ISO8601).", '{"types":{"Admin":["read","write","delete"],"Editor":["read","write"],"Viewer":["read"]},"fields":{"id":"UUID","email":{"unique":true},"name":"string","created_at":"ISO8601"}}'),
    ("hard", "Extract nested company", "TechNova: Engineering(Sarah Chen, 45, teams: Backend/Mike, Frontend/Lisa, DevOps/Tom), Marketing(James Wright, 12), Operations(Priya Patel, 8).", '{"departments":[{"name":"Engineering","head":"Sarah Chen","count":45,"teams":[{"name":"Backend","lead":"Mike"},{"name":"Frontend","lead":"Lisa"},{"name":"DevOps","lead":"Tom"}]},{"name":"Marketing","head":"James Wright","count":12},{"name":"Operations","head":"Priya Patel","count":8}]}'),
    ("hard", "Extract error response", '{"error":{"code":422,"type":"validation","message":"Invalid email","field":"email","details":{"expected":"user@domain.tld","received":"bad"}}}', '{"error":{"code":422,"type":"validation","message":"Invalid email","field":"email","details":{"expected":"user@domain.tld","received":"bad"}}}'),
    ("hard", "Extract API pagination", "Response: page 3 of 50, per_page 25, total 1247 items. Links: first /api?page=1, prev /api?page=2, next /api?page=4, last /api?page=50", '{"page":3,"pages":50,"per_page":25,"total":1247}'),
    ("hard", "Extract matrix data", "Grade distribution: A:15(25%), B:22(37%), C:14(23%), D:6(10%), F:3(5%). Total: 60.", '{"distribution":[{"grade":"A","count":15,"pct":25},{"grade":"B","count":22,"pct":37},{"grade":"C","count":14,"pct":23}],"total":60}'),
    ("hard", "Extract YAML config", "server: host 0.0.0.0, port 8080, workers 4. db: url postgres://localhost/mydb, pool 10. log: level INFO, format json.", '{"server":{"host":"0.0.0.0","port":8080,"workers":4},"database":{"url":"postgres://localhost/mydb","pool_size":10},"logging":{"level":"INFO","format":"json"}}'),
    ("hard", "Extract dependency tree", "deps: react ^18, axios ^1.6. devDeps: typescript ^5.3, vite ^5.0. peer: react >=16.", '{"dependencies":{"react":"^18","axios":"^1.6"},"devDependencies":{"typescript":"^5.3","vite":"^5.0"},"peerDependencies":{"react":">=16"}}'),
]
for diff, intent, ctx, answer in json_extracts:
    add("json", diff, intent, ctx, answer, ["valid JSON", "exact fields"])

# Multi-turn JSON
json_mt = [
    ("medium", "Build API response format", [
        {"user": "Design a JSON API response format for a paginated list endpoint.", "assistant": '{"status":200,"data":[...],"pagination":{"page":1,"per_page":20,"total":100,"pages":5},"links":{"next":"/api/items?page=2"}}'},
        {"user": "Add error handling format.", "assistant": '{"status":422,"error":{"code":"validation_error","message":"Invalid email","field":"email","details":{"expected":"user@domain.com"}}} Keep status at top level for consistency.'},
        {"user": "What about rate limit headers?", "assistant": "Use HTTP headers, not body: X-RateLimit-Limit: 100, X-RateLimit-Remaining: 42, X-RateLimit-Reset: 1609459200. Return 429 with Retry-After header when exceeded."}
    ]),
    ("medium", "Design database schema", [
        {"user": "Design a simple user + posts schema in JSON.", "assistant": '{"users":{"fields":{"id":"UUID","email":"string(unique)","name":"string","created_at":"datetime"}},"posts":{"fields":{"id":"UUID","user_id":"UUID(FK)","title":"string","content":"text","published_at":"datetime"}}}'},
        {"user": "Add comments.", "assistant": '{"comments":{"fields":{"id":"UUID","post_id":"UUID(FK)","user_id":"UUID(FK)","content":"text","created_at":"datetime"}},"indexes":["post_id","user_id"]}'},
    ]),
]
for diff, intent, turns in json_mt:
    add_mt(diff, intent, turns)

# ── REASONING (100) ──────────────────────────────────────
reasoning = [
    ("easy", "Train at 60mph for 2.5 hours?", "", "60 × 2.5 = 150 miles."),
    ("easy", "Apples at 1.20 GBP each, cost for 7?", "", "7 × 1.20 = 8.40 GBP."),
    ("easy", "Today Wednesday, what day in 10 days?", "", "10 mod 7 = 3. Wednesday + 3 = Saturday."),
    ("easy", "15% of 240?", "", "0.15 × 240 = 36."),
    ("easy", "Next: 2, 6, 12, 20, 30, ?", "", "Differences: 4,6,8,10,12. Next: 30+12 = 42."),
    ("easy", "Pizza cut into 8 slices, eat 3. What remains?", "", "5/8 remains."),
    ("easy", "Eggs cost 3.60 GBP for 12. Per egg?", "", "3.60 / 12 = 0.30 GBP per egg."),
    ("easy", "All roses are flowers. Some flowers fade quickly. Do all roses fade?", "", "No. 'Some flowers' doesn't mean all. Can't conclude all roses fade."),
    ("medium", "Chickens and cows: 30 heads, 80 legs.", "", "c + w = 30, 2c + 4w = 80. c = 20 chickens, w = 10 cows."),
    ("medium", "Bill 87.50 GBP, 15% tip, 3 ways.", "", "87.50 × 1.15 = 100.625. Each: 33.54 GBP."),
    ("medium", "Ball dropped 16m, bounces half. After 4?", "", "8, 4, 2, 1. After 4 bounces: 1 meter."),
    ("medium", "20% off then 10% off. Total discount?", "", "0.80 × 0.90 = 0.72. Total discount: 28%."),
    ("medium", "5 machines, 5 widgets, 5 min. 100 machines, 100 widgets?", "", "Each makes 1 widget in 5 min. 100 machines: 5 minutes."),
    ("medium", "Pipe A: 6hrs, Pipe B: 4hrs. Together?", "", "1/6 + 1/4 = 5/12 per hour. Time: 2.4 hours."),
    ("medium", "30 students: 18 football, 15 cricket, 8 both. Neither?", "", "18+15-8 = 25 play something. Neither: 30-25 = 5."),
    ("medium", "Speed: 40mph first half, 60mph second (by distance)?", "", "Harmonic mean: 2×40×60/(40+60) = 48 mph."),
    ("medium", "Speed of sound 343 m/s. Lightning 3 seconds away?", "", "343 × 3 = 1029 meters, about 1 km."),
    ("medium", "Two trains 100mi apart, 60+40mph toward each other.", "", "Combined: 100 mph. 100 miles / 100 mph = 1 hour."),
    ("medium", "A snail climbs 3/day, slips 2/night. 30-foot well.", "", "Net: 1ft/day. Day 28: climbs to 30ft and escapes. 28 days."),
    ("medium", "8 people, 6 hours to build wall. 12 people?", "", "48 person-hours / 12 = 4 hours."),
    ("medium", "Average speed 40 then 60 mph. Total discount.", "", "Not simply average. Harmonic: 2×40×60/100 = 48 mph."),
    ("medium", "9 balls, 1 heavier. Min weighings?", "", "3 weighings. Split 4-4-4, weigh two groups. Then narrow."),
    ("medium", "Water jug: 3L and 5L, measure 4L.", "", "Fill 5L. Pour to 3L (2L left). Empty 3L. Pour 2L to 3L. Fill 5L. Pour to 3L (1L fits). 5L has 4L."),
    ("medium", "Birthday paradox with 23 people?", "", "P ≈ 50.7%. Surprisingly high because you're checking all pairs."),
    ("medium", "Monty Hall: should you switch?", "", "Yes. Switching wins 2/3. Initial pick has 1/3 chance; remaining door gets 2/3."),
    ("medium", "How many times clock hands overlap in 24h?", "", "22 times. 11 per 12 hours (not 12 — between 11 and 1 there's only 1)."),
    ("medium", "100 coins: 99 fair, 1 double-headed. Flip heads. P(other side heads)?", "", "P(double|heads) = 1×0.01 / (0.01 + 0.99×0.5) = 0.01/0.505 ≈ 1.98%."),
    ("medium", "Man 4x son's age. In 20 years, 2x. Ages?", "", "Father = 4x, son = x. 4x+20 = 2(x+20). x=10. Son: 10, Father: 40."),
    ("medium", "How many squares on chessboard?", "", "64(1×1) + 49(2×2) + ... + 1(8×8) = 204 total squares."),
    ("medium", "3 red, 2 blue balls. Draw 2 without replacement. P(both red)?", "", "(3/5)×(2/4) = 6/20 = 3/10 = 30%."),
    ("medium", "Clock at 3:15, angle between hands?", "", "Minute: 90°. Hour: 97.5°. Angle: 7.5 degrees."),
    ("hard", "12 balls, 1 heavier. Min weighings?", "", "3 weighings. Split 3-3-3-3. Weigh 3v3. Then narrow with 1v1."),
    ("hard", "100 doors: toggle every nth. Which open?", "", "Perfect squares: 1,4,9,16,25,36,49,64,81,100. Odd number of factors."),
    ("hard", "25 horses, 5-horse track. Min races for top 3?", "", "7 races. Race 5 groups (5), race winners (6), race top candidates (7)."),
    ("hard", "Prisoner's dilemma iterated 100 rounds?", "", "Both defect every round (Nash equilibrium). Cooperation unravels from last round."),
    ("hard", "Two envelopes paradox. Should you switch?", "", "No net gain. The paradox mixes prior and posterior probabilities incorrectly."),
    ("hard", "If 3 cats catch 3 mice in 3 min, how many cats for 100 mice in 100 min?", "", "Rate: 1 cat catches 1 mouse per 3 min. In 100 min, 1 cat catches ~33. Need 3 cats for 100 mice in 100 min."),
]
for diff, intent, ctx, answer in reasoning:
    add("reasoning", diff, intent, ctx, answer)

# ── DOMAIN-SPECIFIC: GameFAQ + Deobfuscation (125) ──────────────────────────────
# GameFAQ extractions
gamefaq = [
    ("easy", "Extract locations", "In Eldervale, start at Village Square. North to Forest Path. East to Ironspire Keep.", '{"locations":["Village Square","Forest Path","Ironspire Keep"]}'),
    ("easy", "List boss locations", "Crystal Caverns boss: Frost Wyrm (HP 3000, weak fire). Drops Frost Shard.", '{"bosses":[{"name":"Frost Wyrm","hp":3000,"weakness":"fire","drops":["Frost Shard"]}]}'),
    ("easy", "Extract NPC info", "Merchant Brin in Lakeside Village buys fish (5g), sells Fishing Rod (20g), Bait Pack (5g).", '{"npc":{"name":"Merchant Brin","location":"Lakeside Village","sells":[{"item":"Fishing Rod","price":20}]}}'),
    ("easy", "List enemy weaknesses", "Frozen Wastes: Ice Golem (weak fire, immune cold), Frost Wolf (weak lightning).", '{"enemies":[{"name":"Ice Golem","weakness":"fire","immune":"cold"},{"name":"Frost Wolf","weakness":"lightning"}]}'),
    ("easy", "Extract save points", "Ch3: Forest Camp (after bridge), Mountain Shrine (near waterfall), Castle Gatehouse (before boss).", '{"save_points":[{"name":"Forest Camp","near":"bridge"},{"name":"Mountain Shrine","near":"waterfall"},{"name":"Castle Gatehouse","near":"boss"}]}'),
    ("easy", "Extract potion recipes", "Health: 2 Red Herbs + 1 Spring Water. Mana: 2 Blue Herbs + 1 Spring Water.", '{"potions":[{"name":"Health Potion","materials":[{"item":"Red Herb","qty":2},{"item":"Spring Water","qty":1}]}]}'),
    ("easy", "Extract weapon stats", "Iron Sword (atk 15, weight 3), Steel Greatsword (atk 28, weight 8), Shadow Dagger (atk 12, bonus stealth).", '{"weapons":[{"name":"Iron Sword","atk":15,"weight":3},{"name":"Shadow Dagger","atk":12,"bonus":"stealth"}]}'),
    ("medium", "Build quest progression", "Lost Crown: 1) Talk Elder Theron (Oakhaven). 2) Find Ancient Map (Ruined Library). 3) Navigate Whispering Woods. 4) Defeat Shadow Guardian. 5) Return to Theron.", '{"quest":"The Lost Crown","steps":[{"step":1,"action":"Talk Elder Theron","location":"Oakhaven"},{"step":2,"action":"Find Ancient Map","location":"Ruined Library"}]}'),
    ("medium", "Map dungeon layout", "Darkhollow: Entrance -> Hall of Echoes (west: Crypt of Bones, east: Flooded Passage). Crypt has lever opening Flooded Passage gate. Passage -> Boss Chamber.", '{"rooms":[{"name":"Entrance","to":["Hall of Echoes"]},{"name":"Hall of Echoes","to":["Crypt of Bones","Flooded Passage"]}]}'),
    ("medium", "Build area graph", "Overgrown Path -> Mossy Glade, Spider Nest. Mossy Glade -> Fairy Ring (needs Charm). Spider Nest -> Queen's Lair. Fairy Ring -> Sky Garden.", '{"areas":["Overgrown Path","Mossy Glade","Spider Nest","Fairy Ring","Queen Lair","Sky Garden"],"edges":[{"from":"Overgrown Path","to":"Mossy Glade"}]}'),
    ("medium", "Extract skill tree", "Warrior: T1 Slash(5AP), Block(3AP). T2(requires Slash): Whirlwind(8AP), Cleave(6AP). T3(requires Whirlwind): Berserker Rage(15AP).", '{"skills":[{"name":"Slash","tier":1,"cost":"5AP"},{"name":"Whirlwind","tier":2,"requires":"Slash","cost":"8AP"}]}'),
    ("medium", "Extract crafting recipes", "Iron Sword = 2 Iron + 1 Leather. Steel Greatsword = 3 Steel + 2 Leather + 1 Ruby.", '{"recipes":[{"result":"Iron Sword","materials":[{"item":"Iron","qty":2},{"item":"Leather","qty":1}]}]}'),
    ("medium", "Extract boss guide", "Lich King (Crypt of Shadows). P1(100-50%): summons 4 skeletons/30s, weak holy. P2(50-0%): Shadow Bolt 500dmg, invulnerable during summon.", '{"boss":"Lich King","phases":[{"phase":1,"hp":"100-50%","abilities":["summon 4 skeletons/30s"],"weakness":"holy"},{"phase":2,"hp":"50-0%","abilities":["Shadow Bolt 500dmg"]}]}'),
    ("medium", "Extract NPC schedule", "Blacksmith Gareth: Mon-Fri 6-12 Forge, 12-2 Tavern, 2-6 Forge, 6-10 Home. Sat 8-4 Market. Sun Chapel.", '{"npc":"Blacksmith Gareth","schedule":{"weekdays":[{"time":"6-12","location":"Forge"},{"time":"12-2","location":"Tavern"}],"saturday":[{"time":"8-4","location":"Market"}]}}'),
    ("medium", "Extract treasure locations", "Sunken Temple: R1 chest behind pillar (200g). R3 secret wall (Amulet). R5 underwater (Pearl).", '{"treasures":[{"room":1,"item":"200 gold"},{"room":3,"item":"Enchanted Amulet"},{"room":5,"item":"Pearl of Wisdom"}]}'),
    ("medium", "Extract map coordinates", "Village(0,0), Dark Forest(3,5), Crystal Lake(-2,7), Volcano(8,3), Ruins(5,-4), Port(-6,2).", '{"pois":[{"name":"Village","x":0,"y":0},{"name":"Dark Forest","x":3,"y":5},{"name":"Crystal Lake","x":-2,"y":7}]}'),
    ("medium", "Extract quest chain", "Dragon Slayer: 1) Egg Hunt (find 3 eggs). 2) Hatchling (protect 5 days). 3) First Flight (tame dragon). 4) Dragon Rider (defeat Dragon Lord).", '{"chain":"Dragon Slayer","quests":[{"name":"Egg Hunt","task":"find 3 eggs"},{"name":"Dragon Rider","task":"defeat Dragon Lord"}]}'),
    ("medium", "Extract encounter table", "Shadow Marsh: 40% Bog Lurker (easy), 25% Witch Circle (medium), 15% Treasure, 10% Merchant, 5% Will-o-Wisp, 5% Nothing.", '{"encounters":[{"event":"Bog Lurker","chance":40},{"event":"Witch Circle","chance":25},{"event":"Treasure","chance":15}]}'),
    ("hard", "Full fortress extraction", "Thornwall: L1 Armory(rusty key from Captain), Barracks(3 enemies). L2 War Room(Commander Vex, quest Siege Breaker), Prison(captured scout). L3 Throne Room(boss Kael HP5000, 2 phases).", '{"fortress":"Thornwall","levels":[{"level":1,"rooms":[{"name":"Armory","key":"rusty key"}]},{"level":2,"rooms":[{"name":"War Room","npc":"Commander Vex"}]},{"level":3,"rooms":[{"name":"Throne Room","boss":"Kael"}]}]}'),
    ("hard", "Quest dependency graph", "Lost Key -> Sealed Door. Wolf Hunt -> Alpha Wolf, Wolf Pelts. Sealed Door + Alpha Wolf -> Ancient Guardian.", '{"quests":[{"name":"Lost Key","prereqs":[]},{"name":"Sealed Door","prereqs":["Lost Key"]},{"name":"Ancient Guardian","prereqs":["Sealed Door","Alpha Wolf"]}]}'),
    ("hard", "Full walkthrough extraction", "Ch5: Harbor -> Captain Mira (Ship Pass). Sail to Storm Isle. Navigate maze (right, left, straight, right). Find Storm Crystal. Return. Boss: Storm Elemental (weak earth, immune lightning, HP 4000).", '{"chapter":5,"steps":[{"action":"Talk Captain Mira","item":"Ship Pass"},{"action":"Navigate maze","solution":"right, left, straight, right"},{"action":"Defeat Storm Elemental","boss":{"weakness":"earth","hp":4000}}]}'),
    ("hard", "Enchantment system", "Weapons: Fire (+20 dmg, 3 Fire Essence), Ice (slow, 3 Ice Essence), Life Steal (10% heal, 5 Soul Gems). Armor: Fortify HP (+50, 3 Life Essence), Resist Magic (15%, 3 Arcane Dust).", '{"weapons":[{"enchant":"Fire","effect":"+20 dmg","cost":"3 Fire Essence"},{"enchant":"Ice","effect":"slow","cost":"3 Ice Essence"}],"armor":[{"enchant":"Fortify HP","effect":"+50 HP","cost":"3 Life Essence"}]}'),
    ("medium", "Extract faction system", "Factions: Order of Shield (Vex), Shadow Thieves (Whisper), Druid Circle (Moongrove). Levels: Hostile(-100 to -50), Unfriendly, Neutral, Friendly, Exalted(100+).", '{"factions":[{"name":"Order of Shield","quest_giver":"Vex"},{"name":"Shadow Thieves","quest_giver":"Whisper"}],"levels":["Hostile","Unfriendly","Neutral","Friendly","Exalted"]}'),
    ("medium", "Extract mount system", "Horse (+30% speed, 500g), Warhorse (+40% + combat, 2000g), Griffin (flying +60%, needs Exalted Sky Riders, 10000g).", '{"mounts":[{"name":"Horse","speed":"+30%","cost":"500g"},{"name":"Griffin","speed":"+60%","ability":"flying","requires":"Exalted Sky Riders"}]}'),
    ("medium", "Extract pet stats", "Cat (+5% stealth, 50g), Dog (+10% detection, 75g), Hawk (+15% ranged accuracy, 200g).", '{"pets":[{"name":"Cat","bonus":"+5% stealth","cost":"50g"},{"name":"Dog","bonus":"+10% detection","cost":"75g"},{"name":"Hawk","bonus":"+15% ranged accuracy","cost":"200g"}]}'),
]
for diff, intent, ctx, answer in gamefaq:
    add("gamefaq", diff, intent, ctx, answer, ["valid JSON", "extract all entities"])

# Deobfuscation
deobf = [
    ("easy", "Rename variables", "def f(a,b): return a+b", "def add(x, y): return x + y"),
    ("easy", "Explain code", "d = {k:v for k,v in zip(keys, vals)}", "Creates dict from zipped keys and values. Equivalent to dict(zip(keys, vals))."),
    ("easy", "Explain one-liner", "result = any(x > 10 for x in numbers)", "Returns True if any element > 10. Short-circuits on first match."),
    ("easy", "Explain code", "data = sorted(data, key=lambda x: x['name'])", "Sorts list of dicts by the 'name' field alphabetically."),
    ("medium", "Rename and explain", "def x(a):\n    b = []\n    for c in a:\n        if c not in b: b.append(c)\n    return b", "def deduplicate(items):\n    seen = []\n    for item in items:\n        if item not in seen: seen.append(item)\n    return seen\n\nRemoves duplicates preserving order."),
    ("medium", "Simplify control flow", "if x > 0:\n    if x < 10: return True\n    else: return False\nelse: return False", "return 0 < x < 10"),
    ("medium", "Rename variables", "def calc(t, r, n): return t * (1 + r/n)**(n)", "def compound_amount(principal, rate, periods): return principal * (1 + rate/periods)**(periods)"),
    ("medium", "Simplify verbose code", "output = []\nfor x in data:\n    if x is not None:\n        if x > 0: output.append(x * 2)", "output = [x * 2 for x in data if x is not None and x > 0]"),
    ("medium", "Explain decorator", "@property\ndef full_name(self): return f'{self.first} {self.last}'", "Makes full_name accessible as attribute (obj.full_name) instead of method call (obj.full_name())."),
    ("medium", "Explain obfuscated regex", "p = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')", "Email validation regex. Matches: alphanumeric + special chars + @ + domain + dot + 2+ letter TLD."),
    ("medium", "Identify vulnerability", "import pickle\ndata = pickle.loads(user_input)", "Dangerous — pickle can execute arbitrary code on untrusted input. Use json instead."),
    ("medium", "Explain metaclass", "class Singleton(type):\n    _instances = {}\n    def __call__(cls, *a, **kw):\n        if cls not in cls._instances:\n            cls._instances[cls] = super().__call__(*a, **kw)\n        return cls._instances[cls]", "Singleton metaclass. Ensures only one instance of a class ever exists."),
    ("medium", "Explain generator", "def fib():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b", "Infinite Fibonacci generator. Uses yield for lazy evaluation. Call next() or iterate."),
    ("medium", "Simplify try/except", "try: val = d['key']\nexcept KeyError: val = default", "val = d.get('key', default)"),
    ("medium", "Explain async pattern", "async def fetch_all(urls):\n    async with aiohttp.ClientSession() as session:\n        tasks = [fetch(session, url) for url in urls]\n        return await asyncio.gather(*tasks)", "Concurrent HTTP fetching. One session, parallel tasks, gathered results."),
    ("medium", "Explain bitwise ops", "flags = READ | WRITE\nif flags & WRITE: print('writable')", "Bitwise OR to set flags, AND to check. Each permission is a bit position."),
    ("medium", "Explain callback", "def on_complete(f):\n    def wrapper(*a, **k):\n        result = f(*a, **k)\n        log(f'{f.__name__} done: {result}')\n        return result\n    return wrapper", "Decorator that logs function completion and return value."),
    ("hard", "Deobfuscate function", "def _0x3f(a):\n    _0x1 = []; _0x2 = 0\n    while _0x2 < len(a):\n        _0x3 = a[_0x2]; _0x4 = 1\n        while _0x2+_0x4 < len(a) and a[_0x2+_0x4]==_0x3: _0x4 += 1\n        _0x1.append((_0x3,_0x4)); _0x2 += _0x4\n    return _0x1", "def run_length_encode(data):\n    encoded = []; i = 0\n    while i < len(data):\n        char = data[i]; count = 1\n        while i+count < len(data) and data[i+count] == char: count += 1\n        encoded.append((char, count)); i += count\n    return encoded\n\nRun-length encoding."),
    ("hard", "Identify algorithm", "def mystery(arr):\n    for i in range(1, len(arr)):\n        key = arr[i]; j = i-1\n        while j >= 0 and arr[j] > key:\n            arr[j+1] = arr[j]; j -= 1\n        arr[j+1] = key\n    return arr", "Insertion sort. Shifts elements right until correct position. O(n²) average."),
    ("hard", "Decode bit manipulation", "def count_bits(n):\n    c = 0\n    while n: n &= n-1; c += 1\n    return c", "Brian Kernighan's algorithm. n &= n-1 clears lowest set bit each iteration. Counts 1-bits."),
    ("hard", "Deobfuscate data transform", "data = {k: list(v) for k, v in groupby(sorted(items, key=keyfunc), key=keyfunc)}", "Groups items by key. Sorts first (required by groupby), then groups consecutive same-key items into lists."),
    ("hard", "Explain context manager", "with open('f.txt') as f:\n    data = f.read()\nprint(f.closed)", "Opens file, reads contents, auto-closes on exit. f.closed is True after the with block."),
    ("medium", "Rename variables", "def h(d, k, v=None): return d.get(k, v)", "def safe_get(dictionary, key, default=None): return dictionary.get(key, default)"),
    ("medium", "Explain decorator stacking", "@app.route('/api')\n@login_required\ndef endpoint(): ...", "Route registers handler. Login_required checks auth before allowing access. Execution: app.route wraps login_required wraps endpoint."),
    ("medium", "Simplify nested ternary", "x = 'yes' if a > 0 else ('no' if a < 0 else 'maybe')", "if a > 0: result = 'yes'\nelif a < 0: result = 'no'\nelse: result = 'maybe'"),
    ("medium", "Explain obfuscated code", "p = lambda s: print(s[::-1])", "Lambda that reverses and prints a string. Equivalent to: def reverse_print(s): print(s[::-1])"),
    ("hard", "Explain code", "[x for x in range(100) if all(x%i!=0 for i in range(2,x))]", "Generates prime numbers below 100. For each x, checks no integer from 2 to x-1 divides it."),
    ("medium", "Explain pattern", "result = reduce(lambda a,b: {**a,**b}, [dict(zip(range(i*3,(i+1)*3),v)) for i,v in enumerate(chunks)])", "Flattens chunked data into numbered dict. Each chunk of 3 gets sequential keys. Merges via reduce."),
    ("medium", "Explain", "result = list(map(int, filter(str.isdigit, items)))", "Filters items to digit strings, converts each to int. Equivalent: [int(x) for x in items if x.isdigit()]"),
    ("medium", "Explain code", "x = [i for i in range(10) if i % 2 == 0]", "List of even numbers from 0 to 8: [0, 2, 4, 6, 8]."),
]
for item in deobf:
    if len(item) == 4:
        diff, intent, ctx, answer = item
    else:
        diff, intent, answer = item
        ctx = intent
    add("deobfuscation", diff, intent, ctx, answer)

# ── ANTI-SLOP / CONCISION (50) ──────────────────────────────
concision = [
    ("easy", "Explain git in one sentence.", "", "Git is a distributed version control system that tracks changes in files, allowing multiple people to work on the same codebase."),
    ("easy", "What is Docker in under 15 words?", "", "Docker packages apps into portable containers that run consistently on any system."),
    ("easy", "What is a REST API in one sentence?", "", "An interface where clients send HTTP requests to endpoints to create, read, update, or delete resources."),
    ("easy", "Explain SQL in 15 words.", "", "A language for querying and manipulating structured data in relational databases using tables and columns."),
    ("easy", "What is Linux in under 20 words?", "", "An open-source Unix-like OS kernel powering most servers, Android phones, and supercomputers."),
    ("easy", "Explain cookies (web) in 15 words.", "", "Small text files websites store in your browser to remember preferences, sessions, and tracking data."),
    ("easy", "What is a variable in one sentence?", "", "A named container that stores a value in memory so you can reference and modify it later."),
    ("medium", "Explain Kubernetes without jargon.", "", "Kubernetes automatically manages containers — starting them, scaling up or down, and restarting if they crash."),
    ("medium", "What is serverless in 20 words?", "", "You write code, the cloud runs it on demand. No servers to manage, you pay per execution."),
    ("medium", "Explain OAuth to a 10-year-old.", "", "Instead of giving an app your password, you show a special pass from Google that says 'this app can see X'."),
    ("medium", "What is a load balancer in one sentence?", "", "A traffic controller that distributes incoming requests across multiple servers so no single one gets overwhelmed."),
    ("medium", "Explain microservices vs monolith concisely.", "", "Monolith: one big app. Microservices: many small apps talking over a network. Monolith simpler to start; microservices scale better."),
    ("medium", "What is CI/CD in under 25 words?", "", "Automated pipeline: code changes are built, tested, and deployed without manual steps. CI = integrate often. CD = deploy often."),
    ("medium", "Explain technical debt in one sentence.", "", "Shortcuts in code that speed up delivery now but create maintenance headaches later — like a loan with interest."),
    ("medium", "What is a hash function in 20 words?", "", "A function converting any input to a fixed-size string. Same input always gives same output. One-way."),
    ("medium", "Explain distributed systems challenge.", "", "Getting multiple independent computers to agree on shared state while any can fail or lose network connectivity."),
    ("medium", "What is eventual consistency?", "", "If no new updates, all replicas eventually converge to the same value — but not immediately."),
    ("medium", "Explain the CAP theorem without jargon.", "", "When the network splits, you must choose: everyone sees the same data (but some can't respond) or everyone responds (but might see stale data)."),
    ("medium", "What is a data lake vs warehouse?", "", "Lake: raw dumps of any data. Warehouse: cleaned, structured data optimised for SQL queries."),
    ("medium", "Explain Agile in one sentence.", "", "Iterative approach: build in small increments, get feedback frequently, adapt plans based on what you learn."),
    ("medium", "What is a container vs VM?", "", "Container: shares host OS kernel, lightweight, starts in seconds. VM: full guest OS, heavier, better isolation."),
    ("medium", "Explain Git rebase vs merge.", "", "Merge: creates a merge commit combining branches. Rebase: replays commits on top of target, creating linear history."),
    ("medium", "What is a CDN in one sentence?", "", "A network of servers worldwide that caches content closer to users, reducing latency and load on origin."),
    ("medium", "Explain idempotency in 15 words.", "", "Calling an operation multiple times produces the same result as calling it once. Safe to retry."),
    ("medium", "What is an API gateway?", "", "A single entry point routing API requests to backend services, handling auth, rate limiting, and logging."),
    ("medium", "Explain NoSQL in one sentence.", "", "Databases that don't use tables — they store data as documents, key-value pairs, graphs, or columns."),
    ("hard", "What is a vector database?", "", "A database optimised for storing and searching high-dimensional vectors using similarity metrics like cosine distance."),
    ("hard", "Explain the actor model.", "", "Independent actors communicate only by sending messages, never sharing state directly."),
    ("hard", "What is a service mesh?", "", "Infrastructure layer handling service-to-service communication: load balancing, encryption, retries, observability — without code changes."),
    ("hard", "Explain zero-knowledge proofs.", "", "Proving you know something without revealing what you know — like proving you're over 18 without showing your birthdate."),
    ("hard", "What is CRDT?", "", "Conflict-free Replicated Data Types: data structures that can be updated independently on multiple nodes and always converge without coordination."),
    ("medium", "What is infrastructure as code?", "", "Managing infrastructure through version-controlled config files instead of manual setup, enabling reproducible deployments."),
    ("medium", "What is middleware?", "", "Software sitting between request and response, processing each request (auth, logging, rate limiting) before it reaches the handler."),
    ("medium", "Explain the observer pattern.", "", "An object maintains a list of dependents and notifies them automatically when its state changes."),
    ("medium", "What is dependency injection?", "", "Instead of creating dependencies inside a class, they're passed in from outside. Makes testing and swapping implementations easier."),
    ("medium", "What is a singleton?", "", "A pattern ensuring only one instance of a class exists. Usually implemented with a class variable storing the instance."),
    ("medium", "What is a proxy server?", "", "Sits between you and the internet, forwarding requests. Can hide your IP, cache content, or block access."),
    ("medium", "What is event-driven architecture?", "", "Components communicate by producing and consuming events rather than calling each other directly."),
    ("medium", "Explain Big O notation.", "", "Describes how algorithm time/space scales with input size. O(1) constant, O(n) linear, O(n²) quadratic."),
    ("medium", "What is a hash map?", "", "Data structure mapping keys to values using a hash function for O(1) average lookup. Python dict is one."),
    ("medium", "What is a reverse proxy?", "", "A server that sits in front of backend servers, forwarding client requests. Handles SSL, caching, load balancing."),
    ("medium", "Explain the factory pattern.", "", "Creates objects without specifying the exact class. A factory method decides which class to instantiate based on input."),
    ("medium", "What is connection pooling?", "", "Reusing a set of pre-established database connections instead of opening/closing for each query. Major performance win."),
    ("medium", "What is a message queue?", "", "A buffer between producers and consumers of messages. Decouples services, handles spikes, enables async processing."),
    ("medium", "Explain the strategy pattern.", "", "Defines a family of algorithms, encapsulates each one, and makes them interchangeable at runtime."),
    ("medium", "What is blue-green deployment?", "", "Two identical production environments. Switch traffic from blue to green for zero-downtime deploys. Rollback by switching back."),
    ("medium", "What is a circuit breaker?", "", "Prevents cascading failures by stopping calls to a failing service after a threshold, then periodically testing if it recovers."),
    ("medium", "Explain the repository pattern.", "", "Abstracts data access behind an interface. Business logic doesn't know if data comes from DB, API, or in-memory cache."),
    ("medium", "What is eventual consistency?", "", "Guarantee that all replicas converge to the same value if no new updates — but not instantly."),
    ("medium", "What is horizontal vs vertical scaling?", "", "Horizontal: add more machines. Vertical: add more power (CPU, RAM) to existing machine. Horizontal scales further."),
]
for diff, intent, ctx, answer in concision:
    add("concision", diff, intent, ctx, answer, ["under 50 words", "no filler"])

# Shuffle and finalize
random.shuffle(examples)
for i, ex in enumerate(examples):
    ex["id"] = f"bsmagpie_{i+1:04d}"

# Write output
os.makedirs("data/canonical", exist_ok=True)
output_path = "data/canonical/bilawal_smol_magpie_v1_canonical.jsonl"
with open(output_path, "w") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

# Print stats
from collections import Counter
domains = Counter(ex["domain"] for ex in examples)
diffs = Counter(ex["difficulty"] for ex in examples)
print(f"Generated {len(examples)} examples -> {output_path}")
print("Domain distribution:")
for d, c in sorted(domains.items()):
    pct = c / len(examples) * 100
    print(f"  {d}: {c} ({pct:.1f}%)")
print("Difficulty distribution:")
for d, c in sorted(diffs.items()):
    pct = c / len(examples) * 100
    print(f"  {d}: {c} ({pct:.1f}%)")
