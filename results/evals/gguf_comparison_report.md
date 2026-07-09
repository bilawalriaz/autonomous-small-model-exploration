# GGUF Model Comparison Report: Base vs SFT

This report compares the output characteristics of the base `LiquidAI/LFM2.5-1.2B-Instruct` model and the fine-tuned SFT version (`lfm25_12b_instruct_sft_q8_strict`) using quantized `Q4_K_M` GGUF files.

## Summary Metrics by Category

| Category | Count | Metric Type | Base GGUF | SFT GGUF | SFT Delta |
| :--- | :---: | :--- | :---: | :---: | :---: |
| coding | 17 | Avg Length (words) | 167.3 | 117.6 | -49.7 |
| concision_antislip | 17 | Avg Length (words) | 58.9 | 63.5 | +4.6 |
| | | Constraint Adherence | 100.0% | 100.0% | +0.0% |
| deobfuscation | 17 | Avg Length (words) | 203.2 | 168.3 | -34.9 |
| factual_qa | 17 | Avg Length (words) | 108.1 | 47.8 | -60.3 |
| | | Factual Accuracy Rate | 5.9% | 23.5% | +17.6% |
| | | Slop Rate (phrases/resp) | 0.06 | 0.00 | -0.06 |
| gamefaq_extraction | 17 | Avg Length (words) | 146.2 | 129.2 | -17.1 |
| | | JSON Validity Rate | 5.9% | 82.4% | +76.5% |
| instruction_following | 17 | Avg Length (words) | 25.5 | 25.4 | -0.2 |
| | | Constraint Adherence | 88.2% | 88.2% | +0.0% |
| json_structured | 17 | Avg Length (words) | 73.8 | 45.4 | -28.4 |
| | | JSON Validity Rate | 76.5% | 76.5% | +0.0% |
| multi_turn | 17 | Avg Length (words) | 285.8 | 206.8 | -79.0 |
| reasoning | 17 | Avg Length (words) | 188.5 | 181.0 | -7.5 |

## Overall Comparison Summary

- **Total Prompts Evaluated**: 153
- **Average Base Length**: 139.7 words
- **Average SFT Length**: 109.4 words
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
```json
[
  {
    "name": "Maria Santos",
    "age": 28,
    "city": "Barcelona"
  }
]
``` [end of text]
```

---

### Example 2: json_structured (eval_021)

**Prompt**:
> Parse this CSV row into JSON: name,city,country
Tokyo,Japan,Asia

**Base GGUF Output**:
```
user
Parse this CSV row into JSON: name,city,country
Tokyo,Japan,Asia
assistant, the parsed JSON would be:

{
  "name": "Tokyo",
  "city": "Japan",
  "country": "Asia"
} [end of text]
```

**SFT GGUF Output**:
```
user
Parse this CSV row into JSON: name,city,country
Tokyo,Japan,Asia
assistant
```json
[
  {
    "name": "Akira Tanaka",
    "city": "Osaka",
    "country": "Japan"
  },
  {
    "name": "Satoshi Kuroda",
    "city": "Kyoto",
    "country": "Japan"
  }
]
``` [end of text]
```

---

### Example 3: json_structured (eval_034)

**Prompt**:
> Extract: 'Meeting at 3pm in Room 4B, attendees: Alice, Bob, Carol. Topic: Q4 budget review. Duration: 1 hour.'

**Base GGUF Output**:
```
user
Extract: 'Meeting at 3pm in Room 4B, attendees: Alice, Bob, Carol. Topic: Q4 budget review. Duration: 1 hour.'
assistantlawyer. I will extract the key details from your text and present them clearly:

- **Event:** Meeting  
- **Time:** 3pm  
- **Location:** Room 4B  
- **Attendees:** Alice, Bob, Carol  
- **Topic:** Q4 budget review  
- **Duration:** 1 hour  

Let me know if you need this formatted differently or require further details! [end of text]
```

**SFT GGUF Output**:
```
user
Extract: 'Meeting at 3pm in Room 4B, attendees: Alice, Bob, Carol. Topic: Q4 budget review. Duration: 1 hour.'
assistant
{
  "extracted_data": [
    {
      "title": "Meeting at 3pm in Room 4B",
      "attendees": ["Alice", "Bob", "Carol"],
      "topic": "Q4 budget review",
      "duration": "1 hour"
    }
  ]
} [end of text]
```

---

### Example 4: gamefaq_extraction (eval_037)

**Prompt**:
> Extract boss data:

Boss: Ice Titan (Frozen Throne Room). HP: 8000. Phase 1: Ground slam (AoE, 200dmg), Frost breath (cone, 300dmg). Phase 2 (below 40% HP): Adds Ice Pillar summon, gains damage reduction. Drops: Titan's Heart (100%), Frost Crown (30%).

**Base GGUF Output**:
```
user
Extract boss data:

Boss: Ice Titan (Frozen Throne Room). HP: 8000. Phase 1: Ground slam (AoE, 200dmg), Frost breath (cone, 300dmg). Phase 2 (below 40% HP): Adds Ice Pillar summon, gains damage reduction. Drops: Titan's Heart (100%), Frost Crown (30%).
assistant, boss data extracted!

**Boss:** Ice Titan  
**Location:** Frozen Throne Room  
**HP:** 8000  

---

### Phase 1 Attacks:
- **Ground Slam:** AoE damage of 200dmg  
- **Frost Breath:** Cone damage of 300dmg (cone effect)

---

### Phase 2 Behavior:
- **Below 40% HP:** Summons Ice Pillar summon
- **Damage Reduction:** Gains damage reduction
- **Drops:**
  - Titan's Heart (100%)
  - Frost Crown (30%)

Let me know if you'd like this formatted for a table or JSON! [end of text]
```

**SFT GGUF Output**:
```
user
Extract boss data:

Boss: Ice Titan (Frozen Throne Room). HP: 8000. Phase 1: Ground slam (AoE, 200dmg), Frost breath (cone, 300dmg). Phase 2 (below 40% HP): Adds Ice Pillar summon, gains damage reduction. Drops: Titan's Heart (100%), Frost Crown (30%).
assistant
{
  "boss_name": "Ice Titan",
  "base_hp": 8000,
  "phase_1_damage": [
    "Ground slam (AoE, 200dmg)",
    "Frost breath (cone, 300dmg)"
  ],
  "phase_2_damage": [
    {
      "damage_reduction": "Ice Pillar summon",
      "drop": "Titan's Heart (100%)",
      "drop_type": "Weapon"
    },
    {
      "damage_reduction": "Frost Crown (30%)",
      "drop": "Frost Crown (30%)",
      "drop_type": "Weapon"
    }
  ]
} [end of text]
```

---

### Example 5: gamefaq_extraction (eval_038)

**Prompt**:
> Extract quest rewards:

Quest: The Iron Gauntlet. Requirements: Level 15+, completed 'Steel Resolve'. Rewards: 500 XP, 200 Gold, Iron Gauntlets (gloves, +15 defense, +5 strength), unlock access to the Arena.

**Base GGUF Output**:
```
user
Extract quest rewards:

Quest: The Iron Gauntlet. Requirements: Level 15+, completed 'Steel Resolve'. Rewards: 500 XP, 200 Gold, Iron Gauntlets (gloves, +15 defense, +5 strength), unlock access to the Arena.
assistant, it looks like you're describing a reward structure for an in-game quest or achievement! Here's a breakdown of what you've provided:

**Quest:** The Iron Gauntlet  
**Requirements:** Level 15+, completed 'Steel Resolve'  
**Rewards:**  
- 500 XP  
- 200 Gold  
- Iron Gauntlets (gloves, +15 defense, +5 strength)  
- Unlock access to the Arena  

This sounds like a reward for completing a challenging task or reaching a certain level in your game. Let me know if you'd like help formatting this into a JSON object, a table, or something else! [end of text]
```

**SFT GGUF Output**:
```
user
Extract quest rewards:

Quest: The Iron Gauntlet. Requirements: Level 15+, completed 'Steel Resolve'. Rewards: 500 XP, 200 Gold, Iron Gauntlets (gloves, +15 defense, +5 strength), unlock access to the Arena.
assistant
{
  "quest_name": "The Iron Gauntlet",
  "rewards": [
    {
      "type": "XP",
      "amount": 500,
      "currency_type": "gold",
      "description": "Earned after completing the quest."
    },
    {
      "type": "Gold",
      "amount": 200,
      "currency_type": "gold",
      "description": "Currency earned from the quest rewards."
    },
    {
      "type": "Iron Gauntlets (gloves)",
      "amount": 1,
      "currency_type": "iron_gauntlet",
      "description": "Gloves that grant +15 defense and +5 strength to your character."
    },
    {
      "type": "Iron Gauntlets (accessory)",
      "amount": 1,
      "currency_type": "gold",
      "description": "Access to the Arena for combat encounters."
    }
  ]
} [end of text]
```

---
