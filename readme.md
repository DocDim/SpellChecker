## Ternary Search Tree (TST) Spell Checker  
**A Python implementation of a spell checker using Ternary Search Trees (TST) for efficient word storage and retrieval, with performance comparisons against list-based search.**

###  Objective  
Design a spell checker that:  
1. Uses **binary search on a sorted list**.  
2. Uses **Ternary Search Trees (TST)** with radix search.  
3. Compares memory/runtime performance.  
4. Suggests corrections for misspelled words.  
 

### Requirements  
- Python 3.x  
- Dependencies:  
  ```bash
  pip install memory_profiler

  ```
  
###  Implementation  

####  Data Structures  
| **Approach**       | **Data Structure** | **Search Algorithm**  | **Key Functions**                     |  
|--------------------|--------------------|-----------------------|---------------------------------------|  
| **List-Based**     | Sorted List        | Binary Search         | `binary_search()`, `spell_checker_list()` |  
| **TST-Based**      | Ternary Search Tree| Radix Search          | `tst_insert()`, `spell_checker_tst()` |  

####  Key Features  
- **Dictionary Loading**: Reads `dictionary.txt` into either a list or TST.  
- **Spell Checking**: Flags misspelled words in `input.txt` with line numbers.  
- **Suggestions**: Recommends top 3 corrections using **Levenshtein distance**.  
- **Performance Metrics**: Tracks memory usage (`tracemalloc`) and runtime (`time` module).  
- **Punctuation Handling**: Ignores common punctuation marks during spell checking. 


### Usage  

#### 1. Prepare files  
   - Place `dictionary.txt` and test files (e.g., `random1000words.txt`) in `data/`.  
   - Update paths in `main()` if needed.   

#### 2. Run the Spell Checker  
```bash
python spell_checker.py
```

#### Output  
- **Memory/Runtime Metrics**: Comparison between TST and list-based methods.  
- **Misspelled Words**: Printed by line number with suggested corrections.  

### Example  
```python
## Sample dictionary.txt
apple
banana
cherry

# #Sample test.txt
I ate an appl and a bannana.

## Output
Line #: 1 - Misspelled words: appl, bannana
Misspelled word: appl - Suggestions: apple, apply, app
Misspelled word: bannana - Suggestions: banana, bandana, banner
```

### Project Structure  
```
tst-spell-checker/
├── data/                   # Input files (dictionary, test text)
├── spell_checker.py        # Main implementation
└── README.md               # This file
```

### Key Functions  
| Function | Description |
|----------|-------------|
| `tst_insert()` | Inserts a word into the TST. |
| `tst_search()` | Searches for a word in the TST. |
| `spell_checker_tst()` | Spell checks a file using TST. |
| `spell_checker_list()` | Spell checks using binary search on a sorted list. |
| `compute_words_distance()` | Calculates Levenshtein distance between words. |
| `spell_suggestor()` | Suggest the top 3 similar words from a given dictionary. |

### Performance Evaluation  
Metrics tracked:  
- **Memory Usage**: Measured using `tracemalloc`.  
- **Runtime**: Compared for TST vs. list operations.  

---

#### Customization Notes:  
1. **File Paths**: Update `dic_file_path` and `test_file_path` in `main()` to match your local paths.  
  

###  Author
*Miltiade D. Tchifou*
