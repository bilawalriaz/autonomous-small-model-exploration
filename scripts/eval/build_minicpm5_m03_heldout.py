#!/usr/bin/env python3
"""Create the frozen, deterministic M03 held-out programmatic evaluation set."""
import json
from pathlib import Path

rows = []
for i, (a, b, c) in enumerate([(14,7,3),(23,5,4),(18,6,2),(31,8,7),(27,9,5),(45,6,3),(16,11,2),(39,4,6),(28,7,4),(52,3,8),(34,5,9),(61,2,7)]):
    rows.append({"eval_id":f"m03_math_{i:02}","category":"math","prompt":f"Compute ({a} + {b}) * {c}. Reply with only the integer.","expected":str((a+b)*c)})
for i, (name, age, active) in enumerate([("Ada",31,True),("Bo",22,False),("Cy",44,True),("Di",19,False),("Eve",28,True),("Fox",35,False),("Gia",41,True),("Hal",26,False),("Ivy",33,True),("Jay",24,False),("Kai",38,True),("Lux",21,False)]):
    expected={"name":name,"age":age,"active":active}
    rows.append({"eval_id":f"m03_json_{i:02}","category":"json","prompt":f"Return only valid JSON with exactly these keys: name, age, active. Values: name={name}; age={age}; active={str(active).lower()}.","expected":expected})
for i, (expr, value) in enumerate([("len('atlas')",5),("sum([2, 4, 6])",12),("'xy' * 3",'xyxyxy'),("max([3, 9, 4])",9),("sorted([3, 1, 2])[-1]",3),("len({'a': 1, 'b': 2})",2),("7 // 2",3),("'MiNi'.lower()",'mini'),("abs(-13)",13),("min([8, 5, 6])",5),("'abc'.replace('b', 'Z')",'aZc'),("2 ** 5",32)]):
    rows.append({"eval_id":f"m03_code_{i:02}","category":"code","prompt":f"What does this Python expression evaluate to? Reply with only the exact result.\n{expr}","expected":str(value)})
for i, (name, arg, value) in enumerate([("lookup","id","17"),("ping","host","db"),("read","path","/tmp/a"),("search","q","atlas"),("delete","id","9"),("deploy","env","staging"),("open","file","note.txt"),("count","kind","task"),("echo","text","hello"),("fetch","url","https://x"),("set","key","mode"),("close","id","3")]):
    rows.append({"eval_id":f"m03_tool_{i:02}","category":"tool","prompt":f"Return exactly one XML function call and nothing else: <function name=\"{name}\"><param name=\"{arg}\">{value}</param></function>","expected":{"name":name,"arg":arg,"value":value}})
out=Path("data/eval/minicpm5_m03_heldout.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(json.dumps(r)+"\n" for r in rows))
print(f"wrote {len(rows)} rows to {out}")
