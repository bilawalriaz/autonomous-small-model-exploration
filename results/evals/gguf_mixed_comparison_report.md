# GGUF Model Comparison Report: Base vs SFT

This report compares the output characteristics of the base `LiquidAI/LFM2.5-1.2B-Instruct` model and the fine-tuned SFT version (`lfm25_12b_instruct_sft_q8_strict`) using quantized `Q4_K_M` GGUF files.

## Summary Metrics by Category

| Category | Count | Metric Type | Base GGUF | SFT GGUF | SFT Delta |
| :--- | :---: | :--- | :---: | :---: | :---: |
| coding | 17 | Avg Length (words) | 167.3 | 127.1 | -40.2 |
| concision_antislip | 17 | Avg Length (words) | 58.9 | 64.7 | +5.8 |
| | | Constraint Adherence | 100.0% | 100.0% | +0.0% |
| deobfuscation | 17 | Avg Length (words) | 203.2 | 163.4 | -39.8 |
| factual_qa | 17 | Avg Length (words) | 108.1 | 33.8 | -74.4 |
| | | Factual Accuracy Rate | 5.9% | 11.8% | +5.9% |
| | | Slop Rate (phrases/resp) | 0.06 | 0.00 | -0.06 |
| gamefaq_extraction | 17 | Avg Length (words) | 146.2 | 126.0 | -20.2 |
| | | JSON Validity Rate | 5.9% | 11.8% | +5.9% |
| instruction_following | 17 | Avg Length (words) | 25.5 | 26.8 | +1.3 |
| | | Constraint Adherence | 88.2% | 88.2% | +0.0% |
| json_structured | 17 | Avg Length (words) | 73.8 | 67.7 | -6.1 |
| | | JSON Validity Rate | 76.5% | 52.9% | -23.5% |
| multi_turn | 17 | Avg Length (words) | 285.8 | 220.1 | -65.7 |
| reasoning | 17 | Avg Length (words) | 188.5 | 91.3 | -97.2 |

## Overall Comparison Summary

- **Total Prompts Evaluated**: 153
- **Average Base Length**: 139.7 words
- **Average SFT Length**: 102.3 words
- **Total Base Assistant Slop Phrases**: 1
- **Total SFT Assistant Slop Phrases**: 0

## Qualitative Differences & Examples

### Example 1: instruction_following (eval_016)

**Prompt**:
> Output the first 5 Fibonacci numbers separated by spaces.

**Base GGUF Output**:
```
user
Output the first 5 Fibonacci numbers separated by spaces.
assistant
Here are the first 5 Fibonacci numbers:

2 1 3 5 8 [end of text]
```

**SFT GGUF Output**:
```
user
Output the first 5 Fibonacci numbers separated by spaces.
assistant
Here is a simple Python script that generates and prints the first 5 Fibonacci numbers:

```python
def fibonacci(n):
    sequence = [0, 1]
    while len(sequence) < n:
        next_number = sequence[-1] + sequence[-2]
        sequence.append(next_number)
    return sequence[:n]

print(fibonacci(5))
```
This script defines a function `fibonacci(n)` that generates the Fibonacci sequence up to the nth number. It then prints out the first 5 numbers in this sequence. [end of text]
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
Here is a Python solution using the json library to parse the string and extract the required information.

```python
import json

text = 'Maria Santos, 28 years old, lives in Barcelona'
data = json.loads(text)

name = data['name']
age = data['age']
city = data['city']

print({'Name': name, 'Age': age, 'City': city})
```

This script will output:

```
{'Name': 'Maria Santos', 'Age': 28, 'City': 'Barcelona'}
``` [end of text]
```

---

### Example 3: json_structured (eval_021)

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
Here is a Python solution using the `csv` and `json` libraries.

```python
import csv
import json

# Read from file
with open('input.csv', 'r') as f:
    reader = csv.DictReader(f)
    
    # Parse each row into JSON
    for row in reader:
        print(row)
        
# Write to a new CSV file
with open('output.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name', 'city', 'country'])
    
    # Write header
    writer.writeheader()
    
    # Parse each row into JSON and write to CSV file
    for row in reader:
        json_row = dict(row)
        writer.writerow(json_row)
```

This script reads a CSV file, parses its rows into Python dictionaries (which are equivalent to lists of key-value pairs), and writes these dictionaries back into a new CSV file. The `csv.DictReader` class is used to automatically parse the CSV data into dictionary format based on the header row in the input CSV file, while the `csv.DictWriter` class is used to write the Python dictionaries back into a CSV file with the same structure as the input.

Please replace 'input.csv' and 'output.csv' with your actual file names if they are different. [end of text]
```

---

### Example 4: json_structured (eval_022)

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
      "value": "apple"
    },
    {
      "id": 2,
      "value": "banana"
    },
    {
      "id": 3,
      "value": "cherry"
    }
  ]
} [end of text]
```

---

### Example 5: json_structured (eval_028)

**Prompt**:
> Extract: 'Temperature 72F, Humidity 45%, Wind 12mph NW, Pressure 1013hPa' into JSON.

**Base GGUF Output**:
```
user
Extract: 'Temperature 72F, Humidity 45%, Wind 12mph NW, Pressure 1013hPa' into JSON.
assistant, Here is the extracted data in JSON format:

```json
{
  "temperature": "72F",
  "humidity": "45%",
  "wind_speed": "12mph",
  "wind_direction": "NW",
  "pressure": "1013hPa"
}
``` [end of text]
```

**SFT GGUF Output**:
```
user
Extract: 'Temperature 72F, Humidity 45%, Wind 12mph NW, Pressure 1013hPa' into JSON.
assistant
```python
import json

data = {
    "temperature": "72F",
    "humidity": "45%",
    "wind_speed": "12mph",
    "pressure": "1013hPa"
}

json_data = json.dumps(data)
print(json_data)
```

Output:
```
{"temperature": "72F", "humidity": "45%", "wind_speed": "12mph", "pressure": "1013hPa"}
``` [end of text]
```

---
