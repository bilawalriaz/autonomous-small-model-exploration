# GGUF Model Comparison Report: Base vs SFT

This report compares the output characteristics of the base `LiquidAI/LFM2.5-1.2B-Instruct` model and the fine-tuned SFT version (`lfm25_12b_instruct_sft_q8_strict`) using quantized `Q4_K_M` GGUF files.

## Summary Metrics by Category

| Category | Count | Metric Type | Base GGUF | SFT GGUF | SFT Delta |
| :--- | :---: | :--- | :---: | :---: | :---: |
| coding | 17 | Avg Length (words) | 167.3 | 128.1 | -39.2 |
| concision_antislip | 17 | Avg Length (words) | 58.9 | 57.6 | -1.2 |
| | | Constraint Adherence | 100.0% | 100.0% | +0.0% |
| deobfuscation | 17 | Avg Length (words) | 203.2 | 163.4 | -39.8 |
| factual_qa | 17 | Avg Length (words) | 108.1 | 42.4 | -65.7 |
| | | Factual Accuracy Rate | 5.9% | 23.5% | +17.6% |
| | | Slop Rate (phrases/resp) | 0.06 | 0.00 | -0.06 |
| gamefaq_extraction | 17 | Avg Length (words) | 146.2 | 141.0 | -5.2 |
| | | JSON Validity Rate | 5.9% | 76.5% | +70.6% |
| instruction_following | 17 | Avg Length (words) | 25.5 | 23.6 | -1.9 |
| | | Constraint Adherence | 88.2% | 88.2% | +0.0% |
| json_structured | 17 | Avg Length (words) | 73.8 | 42.8 | -31.0 |
| | | JSON Validity Rate | 76.5% | 82.4% | +5.9% |
| multi_turn | 17 | Avg Length (words) | 285.8 | 179.5 | -106.3 |
| reasoning | 17 | Avg Length (words) | 188.5 | 182.0 | -6.5 |

## Overall Comparison Summary

- **Total Prompts Evaluated**: 153
- **Average Base Length**: 139.7 words
- **Average SFT Length**: 106.7 words
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
assistant
```json
[
  {
    "id": 123,
    "value": "42"
  },
  {
    "id": 456,
    "value": "99"
  },
  {
    "id": 789,
    "value": "100"
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
      "meeting_details": {
        "date_time": "3pm",
        "location": "Room 4B",
        "attendees": ["Alice", "Bob", "Carol"],
        "topic": "Q4 budget review",
        "duration_hours": 1
      }
    }
  ]
} [end of text]
```

---

### Example 4: gamefaq_extraction (eval_035)

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
assistant
```json
{
  "locations": [
    {
      "name": "Willowbrook Village",
      "description": "The village where Elder Maeve is located."
    },
    {
      "name": "Darkwood Trail",
      "description": "A dark forest path leading north from the village."
    },
    {
      "name": "Crystal Cave",
      "description": "An underground cave containing a glowing crystal."
    }
  ],
  "npc_collected": [
    {
      "name": "Elder Maeve (quest giver)",
      "role": "NPC",
      "interaction_type": "conversation"
    },
    {
      "name": "Miner NPC",
      "role": "NPC",
      "interaction_type": "commerce"
    }
  ]
}
``` [end of text]
```

---

### Example 5: gamefaq_extraction (eval_037)

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
  "boss": "Ice Titan",
  "name": "Ice Titan",
  "hp": 8000,
  "phase1_damage": [
    "Ground slam (AoE, 200dmg)",
    "Frost breath (cone, 300dmg)"
  ],
  "phase2_actions": [
    {
      "action_name": "Adds Ice Pillar summon",
      "damage_reduction": "30%",
      "drop_details": [
        {
          "item_name": "Ice Crown",
          "quantity": 1,
          "weight": 0.5
        }
      ]
    },
    {
      "action_name": "Drops: Titan's Heart (100%)",
      "damage_reduction": "30%",
      "drop_details": [
        {
          "item_name": "Frost Crown (30%)",
          "quantity": 1,
          "weight": 0.3
        }
      ]
    },
    {
      "action_name": "Drops: Frost Crown (30%)",
      "damage_reduction": "20%",
      "drop_details": [
        {
          "item_name": "Frost Crown (100%)",
          "quantity": 1,
          "weight": 0.2
        }
      ]
    }
  ],
  "phase_2_hp_loss_percent": 40
} [end of text]
```

---
