# GGUF Model Comparison Report: Base vs SFT

This report compares the output characteristics of the base `LiquidAI/LFM2.5-1.2B-Instruct` model and the fine-tuned SFT version (`lfm25_12b_instruct_sft_q8_strict`) using quantized `Q4_K_M` GGUF files.

## Summary Metrics by Category

| Category | Count | Metric Type | Base GGUF | SFT GGUF | SFT Delta |
| :--- | :---: | :--- | :---: | :---: | :---: |
| coding | 17 | Avg Length (words) | 167.3 | 124.6 | -42.7 |
| concision_antislip | 17 | Avg Length (words) | 58.9 | 49.1 | -9.8 |
| | | Constraint Adherence | 100.0% | 100.0% | +0.0% |
| deobfuscation | 17 | Avg Length (words) | 203.2 | 174.4 | -28.8 |
| factual_qa | 17 | Avg Length (words) | 108.1 | 39.7 | -68.4 |
| | | Factual Accuracy Rate | 5.9% | 11.8% | +5.9% |
| | | Slop Rate (phrases/resp) | 0.06 | 0.00 | -0.06 |
| gamefaq_extraction | 17 | Avg Length (words) | 146.2 | 114.9 | -31.3 |
| | | JSON Validity Rate | 5.9% | 52.9% | +47.1% |
| instruction_following | 17 | Avg Length (words) | 25.5 | 21.8 | -3.7 |
| | | Constraint Adherence | 88.2% | 88.2% | +0.0% |
| json_structured | 17 | Avg Length (words) | 73.8 | 40.9 | -32.9 |
| | | JSON Validity Rate | 76.5% | 88.2% | +11.8% |
| multi_turn | 17 | Avg Length (words) | 285.8 | 198.1 | -87.7 |
| reasoning | 17 | Avg Length (words) | 188.5 | 213.1 | +24.5 |

## Overall Comparison Summary

- **Total Prompts Evaluated**: 153
- **Average Base Length**: 139.7 words
- **Average SFT Length**: 108.5 words
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
    "id": 1,
    "value": "42"
  },
  {
    "id": 2,
    "value": "99"
  },
  {
    "id": 3,
    "value": "100"
  }
]
``` [end of text]
```

---

### Example 3: json_structured (eval_033)

**Prompt**:
> Output the following as compact JSON (no whitespace): A list of 3 colors with hex codes.

**Base GGUF Output**:
```
user
Output the following as compact JSON (no whitespace): A list of 3 colors with hex codes.
assistant
["#FF5733", "#33FF57", "#3357FF"] [end of text]
```

**SFT GGUF Output**:
```
user
Output the following as compact JSON (no whitespace): A list of 3 colors with hex codes.
assistant
{
  "colors": [
    { "color": "#FF5733", "hex_code": "#FF5733" },
    { "color": "#33A2F5", "hex_code": "#33A2F5" },
    { "color": "#E74C3C", "hex_code": "#E74C3C" }
  ]
} [end of text]
```

---

### Example 4: json_structured (eval_034)

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
  "meeting_details": {
    "location": "Room 4B",
    "time": "3pm",
    "attendees": ["Alice", "Bob", "Carol"],
    "topic": "Q4 budget review",
    "duration": "1 hour"
  }
} [end of text]
```

---

### Example 5: gamefaq_extraction (eval_035)

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
      "type": "village"
    },
    {
      "name": "Darkwood Trail",
      "type": "trail"
    },
    {
      "name": "Crystal Cave",
      "type": "cave"
    }
  ],
  "npc_interactions": [
    {
      "npc_type": "Miner",
      "location": "Crystal Cave",
      "action": "sell pickaxes"
    }
  ]
}
``` [end of text]
```

---
