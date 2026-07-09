# GGUF Model Comparison Report: Base vs SFT

This report compares the output characteristics of the base `LiquidAI/LFM2.5-1.2B-Instruct` model and the fine-tuned SFT version (`lfm25_12b_instruct_sft_q8_strict`) using quantized `Q4_K_M` GGUF files.

## Summary Metrics by Category

| Category | Count | Metric Type | Base GGUF | SFT GGUF | SFT Delta |
| :--- | :---: | :--- | :---: | :---: | :---: |
| coding | 17 | Avg Length (words) | 167.3 | 116.6 | -50.7 |
| concision_antislip | 17 | Avg Length (words) | 58.9 | 53.8 | -5.1 |
| | | Constraint Adherence | 100.0% | 100.0% | +0.0% |
| deobfuscation | 17 | Avg Length (words) | 203.2 | 158.9 | -44.3 |
| factual_qa | 17 | Avg Length (words) | 108.1 | 50.9 | -57.2 |
| | | Factual Accuracy Rate | 5.9% | 11.8% | +5.9% |
| | | Slop Rate (phrases/resp) | 0.06 | 0.00 | -0.06 |
| gamefaq_extraction | 17 | Avg Length (words) | 146.2 | 125.8 | -20.5 |
| | | JSON Validity Rate | 5.9% | 23.5% | +17.6% |
| instruction_following | 17 | Avg Length (words) | 25.5 | 21.8 | -3.7 |
| | | Constraint Adherence | 88.2% | 88.2% | +0.0% |
| json_structured | 17 | Avg Length (words) | 73.8 | 43.1 | -30.6 |
| | | JSON Validity Rate | 76.5% | 82.4% | +5.9% |
| multi_turn | 17 | Avg Length (words) | 285.8 | 193.8 | -92.0 |
| reasoning | 17 | Avg Length (words) | 188.5 | 98.9 | -89.6 |

## Overall Comparison Summary

- **Total Prompts Evaluated**: 153
- **Average Base Length**: 139.7 words
- **Average SFT Length**: 96.0 words
- **Total Base Assistant Slop Phrases**: 1
- **Total SFT Assistant Slop Phrases**: 0

## Qualitative Differences & Examples

### Example 1: json_structured (eval_018)

**Prompt**:
> Extract the name, age, and city from this text into JSON: 'Maria Santos, 28 years old, lives in Barcelona'

**Base GGUF Output**:
```
user
Extract the name, age, and city from this text into JSON: 'Maria Santos, 28 years old, lives in Barcelona'
assistant To extract the name, age, and city from the given text, we can parse the sentence and identify the relevant pieces of information.

**Text:**  
'Maria Santos, 28 years old, lives in Barcelona'

From this, we can extract:
- Name: Maria Santos  
- Age: 28  
- City: Barcelona  

Now, converting these into JSON format:

```json
{
  "name": "Maria Santos",
  "age": 28,
  "city": "Barcelona"
}
```

Let me know if you'd like this in a different format! [end of text]
```

**SFT GGUF Output**:
```
user
Extract the name, age, and city from this text into JSON: 'Maria Santos, 28 years old, lives in Barcelona'
assistant
{
  "name": "Maria Santos",
  "age": 28,
  "city": "Barcelona"
} [end of text]
```

---

### Example 2: json_structured (eval_022)

**Prompt**:
> Output valid JSON only. An array of 3 objects, each with 'id' (1-3) and 'value' (string).

**Base GGUF Output**:
```
user
Output valid JSON only. An array of 3 objects, each with 'id' (1-3) and 'value' (string).
assistant WOW! You've asked for a JSON array containing three objects. Each object should have the structure:

```json
{
  "id": 1,
  "value": "string"
}
```

Here is your requested output:

```json
[
  { "id": 1, "value": "example value"},
  { "id": 2, "value": "another example"},
  { "id": 3, "value": "yet another one" }
]
```

Wait — you said an array of 3 objects. Let me confirm:

You want: an array with 3 objects, each with `id` and `value`. So I'll format it accordingly.

Here's the corrected version:

```json
[
  { "id": 1, "value": "value_one"},
  { "id": 2, "value": "value_two"},
  { "id": 3, "value": "value_three" }
]
```

But since you explicitly said "array of 3 objects", I'll stick to your format:

```json
[
  { "id": 1, "value": "string value 1"},
  { "id": 2, "value": "string value 2"},
  { "id": 3, "value": "string value 3" }
]
```

Let me know if you want them in a different format! [end of text]
```

**SFT GGUF Output**:
```
user
Output valid JSON only. An array of 3 objects, each with 'id' (1-3) and 'value' (string).
assistant Wrong: The array must contain exactly three elements.
{
  "data": [
    {"id": 1, "value": "apple"},
    {"id": 2, "value": "banana"},
    {"id": 3, "value": "cherry"}
  ]
} [end of text]
```

---

### Example 3: gamefaq_extraction (eval_035)

**Prompt**:
> Extract locations and NPCs from this walkthrough text into JSON:

Start at Willowbrook Village. Talk to Elder Maeve (quest giver). Head north through the Darkwood Trail to reach the Crystal Cave. Inside, find the Miner NPC who sells pickaxes.

**Base GGUF Output**:
```
user
Extract locations and NPCs from this walkthrough text into JSON:

Start at Willowbrook Village. Talk to Elder Maeve (quest giver). Head north through the Darkwood Trail to reach the Crystal Cave. Inside, find the Miner NPC who sells pickaxes.
assistant Upload your question and I'll help you extract the locations and NPCs from the walkthrough text into a structured JSON format.

Please provide the walkthrough text if you'd like me to proceed! [end of text]
```

**SFT GGUF Output**:
```
user
Extract locations and NPCs from this walkthrough text into JSON:

Start at Willowbrook Village. Talk to Elder Maeve (quest giver). Head north through the Darkwood Trail to reach the Crystal Cave. Inside, find the Miner NPC who sells pickaxes.
assistantOf course! Here is the extracted information in JSON format:

```json
{
  "locations": [
    {
      "name": "Willowbrook Village",
      "type": "Location"
    },
    {
      "name": "Darkwood Trail",
      "type": "Path"
    },
    {
      "name": "Crystal Cave",
      "type": "Location"
    }
  ],
  "npc_interactions": [
    {
      "location": "Willowbrook Village",
      "npc_type": "Quest Giver",
      "description": "Talk to Elder Maeve (quest giver)"
    },
    {
      "location": "Darkwood Trail",
      "npc_type": "Path",
      "description": "Head north through the Darkwood Trail"
    },
    {
      "location": "Crystal Cave",
      "npc_type": "Location",
      "description": "Inside, find the Miner NPC who sells pickaxes"
    }
  ]
}
```

Let me know if you need any modifications! [end of text]
```

---

### Example 4: gamefaq_extraction (eval_044)

**Prompt**:
> Extract spell data:

Spells: Fireball (Tier 2, 150 dmg, 30 mana, AoE 5m). Ice Lance (Tier 1, 80 dmg, 15 mana, single target, slows). Heal (Tier 2, restores 200 HP, 40 mana). Shield (Tier 3, absorbs 500 dmg, 60 mana, 30s duration).

**Base GGUF Output**:
```
user
Extract spell data:

Spells: Fireball (Tier 2, 150 dmg, 30 mana, AoE 5m). Ice Lance (Tier 1, 80 dmg, 15 mana, single target, slows). Heal (Tier 2, restores 200 HP, 40 mana). Shield (Tier 3, absorbs 500 dmg, 60 mana, 30s duration).
assistant, here are the extracted spell data:

- **Fireball**  
  - Tier: 2  
  - Damage: 150  
  - Mana Cost: 30  
  - Effect: Area of Effect (AoE) for 5 meters  

- **Ice Lance**  
  - Tier: 1  
  - Damage: 80  
  - Mana Cost: 15  
  - Effect: Single target, slows the target  

- **Heal**  
  - Tier: 2  
  - Effect: Restores 200 HP  
  - Mana Cost: 40  

- **Shield**  
  - Tier: 3  
  - Effect: Absorbs 500 damage  
  - Mana Cost: 60  
  - Duration: 30 seconds   [end of text]
```

**SFT GGUF Output**:
```
user
Extract spell data:

Spells: Fireball (Tier 2, 150 dmg, 30 mana, AoE 5m). Ice Lance (Tier 1, 80 dmg, 15 mana, single target, slows). Heal (Tier 2, restores 200 HP, 40 mana). Shield (Tier 3, absorbs 500 dmg, 60 mana, 30s duration).
assistant Below is the extracted spell data in a structured format:

```json
{
  "spells": [
    {
      "name": "Fireball",
      "tier": 2,
      "damage": 150,
      "mana_cost": 30,
      "effect": "AoE",
      "range": "5m"
    },
    {
      "name": "Ice Lance",
      "tier": 1,
      "damage": 80,
      "mana_cost": 15,
      "effect": "Single target",
      "slow": true
    },
    {
      "name": "Heal",
      "tier": 2,
      "damage_restored": 200,
      "mana_cost": 40
    },
    {
      "name": "Shield",
      "tier": 3,
      "damage_absorbed": 500,
      "mana_cost": 60,
      "effect_duration": "30s"
    }
  ]
}
```

Let me know if you need this in a different format or with additional details! [end of text]
```

---

### Example 5: gamefaq_extraction (eval_051)

**Prompt**:
> Extract pet/mount stats:

Pets: Cat (bonus: +5% stealth, cost: 50g), Dog (bonus: +10% detection range, cost: 75g), Hawk (bonus: +15% ranged accuracy, cost: 200g). Mounts: Horse (+30% speed, 500g), War Horse (+30% speed + charge attack, 2000g).

**Base GGUF Output**:
```
user
Extract pet/mount stats:

Pets: Cat (bonus: +5% stealth, cost: 50g), Dog (bonus: +10% detection range, cost: 75g), Hawk (bonus: +15% ranged accuracy, cost: 200g). Mounts: Horse (+30% speed, 500g), War Horse (+30% speed + charge attack, 2000g).
assistant, sure! Here are the extracted pet/mount stats:

**Pets:**
- **Cat:** Bonus of +5% stealth, cost: 50g
- **Dog:** Bonus of +10% detection range, cost: 75g
- **Hawk:** Bonus of +15% ranged accuracy, cost: 200g

**Mounts:**
- **Horse:** (+30% speed, weight: 500g)
- **War Horse:** (+30% speed + charge attack, weight: 2000g)

Let me know if you need more details! [end of text]
```

**SFT GGUF Output**:
```
user
Extract pet/mount stats:

Pets: Cat (bonus: +5% stealth, cost: 50g), Dog (bonus: +10% detection range, cost: 75g), Hawk (bonus: +15% ranged accuracy, cost: 200g). Mounts: Horse (+30% speed, 500g), War Horse (+30% speed + charge attack, 2000g).
assistant APIs to fetch pet/mount stats:

```json
{
  "Pets": [
    {
      "name": "Cat",
      "bonus": "+5% stealth",
      "cost": "50g"
    },
    {
      "name": "Dog",
      "bonus": "+10% detection range",
      "cost": "75g"
    },
    {
      "name": "Hawk",
      "bonus": "+15% ranged accuracy",
      "cost": "200g"
    }
  ],
  "Mounts": [
    {
      "name": "Horse",
      "bonus": "+30% speed",
      "weight": "500g"
    },
    {
      "name": "War Horse",
      "bonus": "+30% speed + charge attack",
      "weight": "2000g"
    }
  ]
}
``` [end of text]
```

---
