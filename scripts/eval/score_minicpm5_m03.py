#!/usr/bin/env python3
"""Programmatically score M03 paired outputs; no model-based judge is used."""
import argparse, json, random, re, statistics, xml.etree.ElementTree as ET
from pathlib import Path

def rows(path): return {r['eval_id']:r for r in map(json.loads, Path(path).read_text().splitlines()) if r}
def json_object(text):
    for m in re.finditer(r'\{.*?\}', text, re.S):
        try: return json.loads(m.group())
        except json.JSONDecodeError: pass
    return None
def score(category, text, expected):
    if category in ('math','code'):
        nums=re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])', text)
        return bool(nums) and nums[-1] == expected
    if category == 'json': return json_object(text) == expected
    match=re.search(r'<function\b.*?</function>', text, re.S)
    if not match: return False
    try:
        root=ET.fromstring(match.group()); child=root.find('param')
        return root.tag=='function' and root.attrib.get('name')==expected['name'] and child is not None and child.attrib.get('name')==expected['arg'] and (child.text or '')==expected['value']
    except ET.ParseError: return False
def ci(deltas, n=20000):
    rng=random.Random(42); means=[]; size=len(deltas)
    for _ in range(n): means.append(sum(deltas[rng.randrange(size)] for _ in range(size))/size)
    means.sort(); return means[int(.025*n)], means[int(.975*n)-1]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--eval-set',required=True); p.add_argument('--base',required=True); p.add_argument('--merged',required=True); p.add_argument('--output',required=True); args=p.parse_args()
    data=rows(args.eval_set); base=rows(args.base); merged=rows(args.merged); result={'integrity':{},'families':{}}
    for family in ('math','json','code','tool'):
        ids=[k for k,v in data.items() if v['category']==family]; bs=[]; ms=[]; pairs=[]
        for k in ids:
            b=score(family,base[k]['generated_response'],data[k]['expected']); m=score(family,merged[k]['generated_response'],data[k]['expected']); bs.append(int(b)); ms.append(int(m)); pairs.append(int(m)-int(b))
        lo,hi=ci(pairs); result['families'][family]={'n':len(ids),'base_rate':sum(bs)/len(ids),'merged_rate':sum(ms)/len(ids),'delta':sum(pairs)/len(ids),'delta_ci95':[lo,hi],'wins':sum(x>0 for x in pairs),'losses':sum(x<0 for x in pairs),'ties':sum(x==0 for x in pairs)}
    for label,run in [('base',base),('merged',merged)]:
        vals=list(run.values()); result['integrity'][label]={'count':len(vals),'nonzero_returncodes':sum(r['returncode']!=0 for r in vals),'prompt_echoes':sum(r['prompt_echo_removed'] for r in vals),'template_residue':sum(r['template_residue_detected'] for r in vals),'empty_outputs':sum(not r['raw_stdout'] for r in vals)}
    result['decision_rule']={'target_ci_lower_bound_gt_0':all(result['families'][f]['delta_ci95'][0]>0 for f in ('math','json')),'control_budget_delta_gte_minus_0_10':all(result['families'][f]['delta_ci95'][0]>=-0.10 for f in ('code','tool'))}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
