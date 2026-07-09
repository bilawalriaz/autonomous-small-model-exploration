# GGUF Model Comparison Report: Base vs SFT

This report compares the output characteristics of the base `LiquidAI/LFM2.5-1.2B-Instruct` model and the fine-tuned SFT version (`lfm25_12b_instruct_sft_q8_strict`) using quantized `Q4_K_M` GGUF files.

## Summary Metrics by Category

| Category | Count | Metric Type | Base GGUF | SFT GGUF | SFT Delta |
| :--- | :---: | :--- | :---: | :---: | :---: |
| coding | 17 | Avg Length (words) | 319.4 | 121.2 | -198.1 |
| | | Slop Rate (phrases/resp) | 0.00 | 0.06 | +0.06 |
| concision_antislip | 17 | Avg Length (words) | 148.8 | 97.9 | -50.8 |
| | | Constraint Adherence | 100.0% | 100.0% | +0.0% |
| deobfuscation | 17 | Avg Length (words) | 363.9 | 151.9 | -212.0 |
| factual_qa | 17 | Avg Length (words) | 135.0 | 121.4 | -13.6 |
| | | Factual Accuracy Rate | 17.6% | 11.8% | -5.9% |
| | | Slop Rate (phrases/resp) | 0.06 | 0.12 | +0.06 |
| gamefaq_extraction | 17 | Avg Length (words) | 232.9 | 117.2 | -115.8 |
| | | JSON Validity Rate | 11.8% | 11.8% | +0.0% |
| instruction_following | 17 | Avg Length (words) | 161.8 | 32.9 | -128.9 |
| | | Constraint Adherence | 88.2% | 88.2% | +0.0% |
| json_structured | 17 | Avg Length (words) | 151.2 | 42.3 | -108.9 |
| | | JSON Validity Rate | 47.1% | 76.5% | +29.4% |
| multi_turn | 17 | Avg Length (words) | 345.5 | 234.6 | -110.9 |
| reasoning | 17 | Avg Length (words) | 369.1 | 115.5 | -253.5 |

## Overall Comparison Summary

- **Total Prompts Evaluated**: 153
- **Average Base Length**: 247.5 words
- **Average SFT Length**: 115.0 words
- **Total Base Assistant Slop Phrases**: 1
- **Total SFT Assistant Slop Phrases**: 3

## Qualitative Differences & Examples

### Example 1: instruction_following (eval_001)

**Prompt**:
> List 5 prime numbers. Output only the numbers, one per line, with no explanation.

**Base GGUF Output**:
```
user
List 5 prime numbers. Output only the numbers, one per line, with no explanation.
assistant<think>
The user wants me to list five prime numbers. Prime numbers are natural numbers greater than 1 that have exactly two distinct positive divisors: 1 and themselves.

I need to generate a list of five such numbers. I'll start with the smallest ones: 2, 3, 5, 7, 11. That gives me five numbers already.

I should ensure they are only one per line as requested. No explanations or additional text.

Let's output exactly:

2
3
5
7
11

That's it. I'll make sure there's no extra content.
</think>

2
3
5
7
11 [end of text]
```

**SFT GGUF Output**:
```
user
List 5 prime numbers. Output only the numbers, one per line, with no explanation.
assistant
2
3
5
7
11 [end of text]
```

---

### Example 2: instruction_following (eval_002)

**Prompt**:
> Convert 'hello world' to uppercase. Do not add any extra text.

**Base GGUF Output**:
```
user
Convert 'hello world' to uppercase. Do not add any extra text.
assistant<think>
The user wants me to convert the string "hello world" to uppercase and then remove any extra text.

I need to process the input: first, I'll take the lowercase version of "hello world", then convert it to uppercase using the `upper()` method. Finally, I'll strip any leading or trailing whitespace since that's what the user didn't want in the output.
</think>

HELLO WORLD [end of text]
```

**SFT GGUF Output**:
```
user
Convert 'hello world' to uppercase. Do not add any extra text.
assistant
HELLO WORLD [end of text]
```

---

### Example 3: json_structured (eval_018)

**Prompt**:
> Extract the name, age, and city from this text into JSON: 'Maria Santos, 28 years old, lives in Barcelona'

**Base GGUF Output**:
```
user
Extract the name, age, and city from this text into JSON: 'Maria Santos, 28 years old, lives in Barcelona'
assistant<think>
We are given a string: 'Maria Santos, 28 years old, lives in Barcelona'
 We need to extract the name, age, and city.
 The format is: first name, then age (with unit), then city.
 We assume that the input is exactly as given.
 Steps:
 1. Split the string by commas? But note: there are spaces after commas. So we can split on commas and then trim each part.
 2. However, the example has a space after comma in 'Maria Santos' so splitting by comma will give two parts: "Maria" and "Santos"
 3. Then for age, it is followed by a space and then the number, so we need to extract the numeric part.
 4. For city, there is no space before the word, but after the comma in the original string? Actually, the original string has: 'Maria Santos, 28 years old, lives in Barcelona'
 So splitting by commas gives: ["Maria", "Santos", "28", "years", "old", "lives", "in", "Barcelona"]
 But note that the first element is a name (without space) and then comma. Then age, then space, then number.
 We can split on commas to get the first two parts: first_name and last_name? Actually, we only have one name in this example.
 Alternatively, we can use regular expression or just take the first word as name and then the next part until space.

 However, the problem says "extract the name, age, and city". So we assume that the input is exactly:
   - First: a string that contains a name (maybe with spaces) but in this example it's without space? Actually, the example has no space between Maria and Santos.
   - Then an integer for age
   - then a word for city

 Let's break down the example string by commas and then trim each part:
  parts = [part1.strip(), part2.strip(), part3.strip(), ...]
 But note: after splitting by comma, we have two elements. So we can do:
  name_part = first element (which is "Maria")
  age_str = next element which is "28"
  city = the rest of the string

 However, if there are more commas, then we might get multiple parts for each person? But the problem says this text only contains one person.

 So let's assume that the input is exactly: 'Maria Santos, 28
```

**SFT GGUF Output**:
```
user
Extract the name, age, and city from this text into JSON: 'Maria Santos, 28 years old, lives in Barcelona'
assistant```json
{
  "name": "Maria",
  "age": 28,
  "city": "Barcelona"
}
``` [end of text]
```

---

### Example 4: json_structured (eval_026)

**Prompt**:
> Parse: 'Order #4521: 3x Widget A ($12.50 each), 1x Widget B ($45.00). Total: $82.50.' Output JSON.

**Base GGUF Output**:
```
user
Parse: 'Order #4521: 3x Widget A ($12.50 each), 1x Widget B ($45.00). Total: $82.50.' Output JSON.
assistant<think>
The user wants me to parse a string containing order details and output it in JSON format.

Let's break down the input:
- "Order #4521: 3x Widget A ($12.50), 1x Widget B ($45.00). Total: $82.50."

I need to extract:
- Order ID: 4521
- Items: 
  - 3x Widget A (price: $12.50)
  - 1x Widget B (price: $45.00)
- Total: $82.50

The output should be in JSON format.

I'll create a JSON object with the required fields:
{
  "order_id": "4521",
  "items": [
    {
      "name": "Widget A"
      "quantity": 3,
      "price": 12.5
    },
    {
      "name": "Widget B"
      "quantity": 1,
      "price": 45.0
    }
  ],
  "total_amount": 82.5
}

I'll ensure the JSON is properly formatted with double quotes for strings and correct indentation.
</think>

```json
{
  "order_id": "4521",
  "items": [
    {
      "name": "Widget A",
      "quantity": 3,
      "price": 12.5
    },
    {
      "name": "Widget B",
      "quantity": 1,
      "price": 45.0
    }
  ],
  "total_amount": 82.5
}
``` [end of text]
```

**SFT GGUF Output**:
```
user
Parse: 'Order #4521: 3x Widget A ($12.50 each), 1x Widget B ($45.00). Total: $82.50.' Output JSON.
assistant```json
{
  "order_id": "4521",
  "items": [
    {
      "name": "Widget A",
      "price": 12.5,
      "quantity": 3
    },
    {
      "name": "Widget B",
      "price": 45.0,
      "quantity": 1
    }
  ],
  "total_amount": 82.50
}
``` [end of text]
```

---

### Example 5: json_structured (eval_028)

**Prompt**:
> Extract: 'Temperature 72F, Humidity 45%, Wind 12mph NW, Pressure 1013hPa' into JSON.

**Base GGUF Output**:
```
user
Extract: 'Temperature 72F, Humidity 45%, Wind 12mph NW, Pressure 1013hPa' into JSON.
assistant<think>
The given text contains a single line of information with temperature, humidity, wind speed, and pressure. The instruction is to extract this data into JSON format.

I will create a JSON object with keys for each field as provided in the text.
</think>

```json
{
  "temperature": 72F,
  "humidity": 45%,
  "wind_speed": 12mph,
  "direction": "NW",
  "pressure": 1013hPa
}
``` [end of text]
```

**SFT GGUF Output**:
```
user
Extract: 'Temperature 72F, Humidity 45%, Wind 12mph NW, Pressure 1013hPa' into JSON.
assistant
```json
{
  "temperature": 72,
  "humidity": 45,
  "wind_speed": 12,
  "direction": "NW",
  "pressure": 1013
}
``` [end of text]
```

---
