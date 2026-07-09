# GGUF Model Comparison Report: Base vs SFT

This report compares the output characteristics of the base `LiquidAI/LFM2.5-1.2B-Instruct` model and the fine-tuned SFT version (`lfm25_12b_instruct_sft_q8_strict`) using quantized `Q4_K_M` GGUF files.

## Summary Metrics by Category

| Category | Count | Metric Type | Base GGUF | SFT GGUF | SFT Delta |
| :--- | :---: | :--- | :---: | :---: | :---: |
| coding | 17 | Avg Length (words) | 167.3 | 138.5 | -28.8 |
| concision_antislip | 17 | Avg Length (words) | 58.9 | 49.5 | -9.4 |
| | | Constraint Adherence | 100.0% | 100.0% | +0.0% |
| deobfuscation | 17 | Avg Length (words) | 203.2 | 207.2 | +4.0 |
| factual_qa | 17 | Avg Length (words) | 108.1 | 63.2 | -44.9 |
| | | Factual Accuracy Rate | 5.9% | 5.9% | +0.0% |
| | | Slop Rate (phrases/resp) | 0.06 | 0.06 | +0.00 |
| gamefaq_extraction | 17 | Avg Length (words) | 146.2 | 139.2 | -7.0 |
| | | JSON Validity Rate | 5.9% | 11.8% | +5.9% |
| instruction_following | 17 | Avg Length (words) | 25.5 | 25.4 | -0.2 |
| | | Constraint Adherence | 88.2% | 88.2% | +0.0% |
| json_structured | 17 | Avg Length (words) | 73.8 | 50.4 | -23.4 |
| | | JSON Validity Rate | 76.5% | 76.5% | +0.0% |
| multi_turn | 17 | Avg Length (words) | 285.8 | 249.9 | -35.9 |
| reasoning | 17 | Avg Length (words) | 188.5 | 184.2 | -4.3 |

## Overall Comparison Summary

- **Total Prompts Evaluated**: 153
- **Average Base Length**: 139.7 words
- **Average SFT Length**: 123.1 words
- **Total Base Assistant Slop Phrases**: 1
- **Total SFT Assistant Slop Phrases**: 1

## Qualitative Differences & Examples

### Example 1: instruction_following (eval_007)

**Prompt**:
> Sort these numbers in descending order: 3, 1, 4, 1, 5, 9. Output as comma-separated list.

**Base GGUF Output**:
```
user
Sort these numbers in descending order: 3, 1, 4, 1, 5, 9. Output as comma-separated list.
assistant
The sorted numbers in descending order are: 9, 5, 4, 3, 1, 1. [end of text]
```

**SFT GGUF Output**:
```
user
Sort these numbers in descending order: 3, 1, 4, 1, 5, 9. Output as comma-separated list.
assistant
To sort the numbers **3, 1, 4, 1, 5, 9** in **descending order**, we first arrange them from largest to smallest:

- Largest: 9  
- Next: 5  
- Then: 4  
- Then: 3 and two 1s (since there are two 1s)  
- Smallest: 1  

So the sorted list in descending order is: **9, 5, 4, 3, 1, 1**

### Final Output:
**9, 5, 4, 3, 1, 1** [end of text]
```

---

### Example 2: json_structured (eval_018)

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
Here is the extracted information in JSON format:

```json
{
  "name": "Maria Santos",
  "age": 28,
  "city": "Barcelona"
}
``` [end of text]
```

---

### Example 3: json_structured (eval_022)

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
{
  "data": [
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
      "value": "27"
    }
  ]
} [end of text]
```

---

### Example 4: json_structured (eval_025)

**Prompt**:
> Create a JSON schema for a book with: title (string, required), author (string, required), isbn (string), pages (integer, min 1).

**Base GGUF Output**:
```
user
Create a JSON schema for a book with: title (string, required), author (string, required), isbn (string), pages (integer, min 1).
assistant The JSON schema you're asking for defines the structure of a **book object** in JSON format. However, your request asks for a **JSON schema**, which is used to define the structure and data types of documents in XML or JSON contexts — not a book itself.

Let's clarify: You want a **JSON schema** that represents a book with the following fields:
- `title` (required string)
- `author` (required string)
- `isbn` (string)
- `pages` (integer, minimum 1)

This is typically used to validate JSON data. Here's how you can define it:

```json
{
  "$schema": "http://json-schema.org/draft/2020-12/schema",
  "title": "bookSchema",
  "type": "object",
  "properties": {
    "title": {
      "type": "string",
      "description": "The title of the book"
    },
    "author": {
      "type": "string",
      "description": "The author's name"
    },
    "isbn": {
      "type": "string",
      "description": "International Standard Book Number"
    },
    "pages": {
      "type": "integer",
      "minimum": 1,
      "description": "Number of pages in the book"
    }
  },
  "required": ["title", "author", "isbn", "pages"]
}
```

### Explanation:
- `$schema`: Refers to the JSON Schema version.
- `type: "object"`: The schema defines an object with properties.
- Each property has a `description` for clarity.
- `required` array specifies which fields must be provided.

Let me know if you want this in another format (like XML or YAML) or need to include additional fields like publication date, publisher, etc. [end of text]
```

**SFT GGUF Output**:
```
user
Create a JSON schema for a book with: title (string, required), author (string, required), isbn (string), pages (integer, min 1).
assistant,

Here’s the JSON schema you requested:

```json
{
  "$schema": "http://json-schema.org/draft/2020-12/schema",
  "title": "Book Schema",
  "type": "object",
  "required": ["title", "author"],
  properties:
    title {
      "type": "string"
    },
    author {
      "type": "string"
    },
    isbn {
      "type": "string"
    },
    pages {
      "type": "integer",
      "minimum": 1
    }
  ]
}
```

**Explanation:**
- `$schema` specifies the JSON Schema version.
- Required fields are `title` and `author`.
- The `pages` field is an integer with a minimum value of 1. [end of text]
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
assistantOf course! Here is the extracted information in JSON format:

```json
{
  "locations": [
    {
      "name": "Willowbrook Village",
      "description": "Start at Willowbrook Village."
    },
    {
      "name": "Darkwood Trail",
      "description": "Head north through the Darkwood Trail to reach the Crystal Cave."
    },
    {
      "name": "Crystal Cave",
      "description": "Inside, find the Miner NPC who sells pickaxes."
    }
  ],
  "npc_interactions": [
    {
      "location": "Willowbrook Village",
      "npc_type": "Elder Maeve (quest giver)",
      "action": "Talk to Elder Maeve"
    },
    {
      "location": "Darkwood Trail",
      "npc_type": "Miner NPC",
      "action": "Find the Miner NPC who sells pickaxes"
    }
  ]
}
```

Let me know if you'd like this formatted differently! [end of text]
```

---
