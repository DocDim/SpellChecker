#import relevant libraries
import time
from memory_profiler import profile
import tracemalloc
import os
import re



#Create the TST data Structure
class Node: 
	def __init__(self, val): 
		self.val = val 
		self.isEndOfString = False
		self.left = None
		self.eq = None
		self.right = None

# Fucntion to insert new word in the TST 		
def tst_insert(root, word):
    """
    Inserts a word into a Ternary Search Tree (TST).

    Parameters:
        root: The root node of the TST.
        word (str): The word to insert.

    Returns:
        The updated root of the TST after insertion.
    """
    # Base case: If the current node is None, create a new node with the first character
    if not root:
        root = Node(word[0])

    # Compare the first character of the word with the current node's value
    if word[0] < root.val:
        # Insert into the left subtree (lexicographically smaller characters)
        root.left = tst_insert(root.left, word)
    elif word[0] > root.val:
        # Insert into the right subtree (lexicographically larger characters)
        root.right = tst_insert(root.right, word)
    else:
        # If more characters remain, insert into the middle subtree
        if len(word) > 1:
            root.eq = tst_insert(root.eq, word[1:])
        else:
            # Mark this node as the end of a valid word
            root.isEndOfString = True

    return root  # Return the updated root after insertion


# Utility function to print out the traversal of a given Ternary Search Tree (TST),
# starting at a given depth.
def traverseTSTUtil(root, buffer, depth):
    """
    Utility function to traverse and print all words stored in a Ternary Search Tree (TST).

    Parameters:
        root: The current node of the TST.
        buffer (list): A list used to store characters while traversing.
        depth (int): The current depth in the tree.

    This function performs an in-order traversal of the TST, printing stored words.
    """
    if root:
        # Recursively traverse the left subtree (lexicographically smaller characters)
        traverseTSTUtil(root.left, buffer, depth)

        # Store the current node's value in the buffer
        buffer[depth] = root.val

        # If this node marks the end of a word, print the accumulated string
        if root.isEndOfString:
            print("".join(buffer[:depth + 1]))  

        # Recursively traverse the middle subtree (continuation of the word)
        traverseTSTUtil(root.eq, buffer, depth + 1)

        # Recursively traverse the right subtree (lexicographically larger characters)
        traverseTSTUtil(root.right, buffer, depth)



# Function use to print out the traversal of a given tst
def traverseTST(root):
    """
    Prints out the traversal of a given Ternary Search Tree (TST).

    Parameters:
        root: The root node of the TST.

    This function initializes a buffer to store characters while traversing 
    the TST and calls a helper function to perform the traversal.
    """
    # Initialize a buffer with a fixed size to store characters during traversal
    buffer = [''] * 50  

    # Call the utility function to perform the traversal and print words
    traverseTSTUtil(root, buffer, 0)
	

# Fucntion to search a word in the TST 
def tst_search(root, word):
    """
    Searches for a word in the Ternary Search Tree (TST).

    Parameters:
        root: The root node of the TST.
        word (str): The word to search for.

    Returns:
        bool: True if the word exists in the TST, False otherwise.
    """
    # Base case: If root is None, the word is not in the TST
    if not root:
        return False

    # Compare the first character of the word with the current node's value
    if word[0] < root.val:
        return tst_search(root.left, word)  # Search in the left subtree
    elif word[0] > root.val:
        return tst_search(root.right, word)  # Search in the right subtree
    else:
        # If more characters remain, search in the middle subtree
        if len(word) > 1:
            return tst_search(root.eq, word[1:])
        else:
            # If at the last character, return whether it marks the end of a word
            return root.isEndOfString

#Function to Loads the content of a text file into a list
def load_text_file_to_list(filename):
    """
    Loads a text file and returns its content as a list of lines.

    Parameters:
        filename (str): The name of the file to read.

    Returns:
        list: A list containing each line of the file as a string (without newline characters).
        None: If the file is not found.
    """
    text_files = []  # Initialize an empty list to store file contents

    try:
        # Open the file in read mode with UTF-8 encoding
        with open(filename, 'r', encoding="utf-8") as file:
            # Read all lines, stripping newline characters
            text_files = [line.rstrip('\n') for line in file]

        return text_files  # Return the list of lines

    except FileNotFoundError:
        # Handle the case where the file is not found
        print(f"Error: File not found: {filename}")
        return None
	
#Function to Loads the content of a text file into a TST   
def load_text_file_to_tst(root, filename):
    """
    Loads a text file and inserts each line into a Ternary Search Tree (TST).

    Parameters:
        root: The root node of the TST.
        filename (str): The name of the file to read.

    Returns:
        The updated root of the TST after inserting all lines.
        None if the file is not found.
    """
    try:
        # Open the file in read mode with UTF-8 encoding
        with open(filename, 'r', encoding="utf-8") as file:
            # Read each line, strip whitespace, and insert it into the TST
            for line in file:
                root = tst_insert(root, line.strip())  # Update root if necessary

        return root  # Return the modified root after processing all lines

    except FileNotFoundError:
        # Handle the case where the file is not found
        print(f"Error: File not found: {filename}")
        return None

# function to perfom  the binary search algorithm  in a list of string
def binary_search(rd_list, target, low=0, high=None):
    """
    Performs binary search on a sorted list to find a target value using a recursive divide-and-conquer strategy.

    Parameters:
        rd_list (list): The sorted list in which to search for the target.
        target: The value to search for.
        low (int): The lower bound index of the search space (default: 0).
        high (int): The upper bound index of the search space (default: len(rd_list) - 1).

    Returns:
        bool: True if the target is found, False otherwise.
    """
    if high is None:
        high = len(rd_list) - 1  # Initialize high on the first call

    # Base case: If search space is empty, return False
    if low > high:
        return False

    # Find the middle index
    middle_index = (low + high) // 2

    # Check if the middle element is the target
    if rd_list[middle_index] == target:
        return True
    elif rd_list[middle_index] < target:
        # Search in the right half
        return binary_search(rd_list, target, middle_index + 1, high)
    else:
        # Search in the left half
        return binary_search(rd_list, target, low, middle_index - 1)

def radix_search(root, word, depth=0):
    """
    Performs a radix search on a Ternary Search Tree (TST).

    Parameters:
        root: The root node of the TST.
        word (str): The word to search for.
        depth (int): The current depth in the tree (default: 0).

    Returns:
        bool: True if the word is found, False otherwise.
    """
    # Base case: If the node is None, the word is not found
    if not root:
        return False

    # Compare the current character of the word with the node's value
    if word[depth] < root.val:
        return radix_search(root.left, word, depth)  # Search left subtree
    elif word[depth] > root.val:
        return radix_search(root.right, word, depth)  # Search right subtree
    else:
        # If at the last character, return whether it marks the end of a word
        if depth == len(word) - 1:
            return root.isEndOfString
        
        # Otherwise, continue searching in the middle subtree
        return radix_search(root.eq, word, depth + 1)

# Function to check if a word contain numerical characters    
def check_numeric(word):
    """
    Checks if a word contains any numerical characters.

    Parameters:
        word (str): The word to check.

    Returns:
        bool: True if the word contains any numeric characters, False otherwise.
    """
    # Check if any character in the word is numeric
    for char in word:
        if char.isnumeric():
            return True  # Return True immediately if a numeric character is found

    return False  # Return False if no numeric characters are found
  


# spell checker function with list
def spell_checker_list(filename, dictList):
    """
    Checks the spelling of words in a file against a provided dictionary list.

    Parameters:
        filename (str): The name of the file to check.
        dictList (list): A list containing all valid words (dictionary).

    Returns:
        dict: A dictionary with line numbers as keys and lists of misspelled words as values.
        None: If the file is not found.
    """
    misspelled = {}  # Dictionary to store misspelled words by line number

    try:
        # Open the file in read mode with UTF-8 encoding
        with open(filename, 'r', encoding="utf-8") as file:
            line_num = 0  # To track the line number

            for line in file:
                line_num += 1  # Increment line number
                misspelled[line_num] = []  # Initialize the list for this line's misspelled words
                
                # Use regex to remove unwanted punctuation and characters
                line = re.sub(r"['’,.\"\-–;_:()!?\[\]{}]", " ", line)  # Replace common punctuation with space
                
                # Split the line into words and check each word
                for word in line.split():
                    word = word.lower()  # Normalize to lowercase
                    # If the word is not in the dictionary and it's not a number, add it to misspelled
                    if not binary_search(dictList, word) and not check_numeric(word):
                        misspelled[line_num].append(word)

        # Return the dictionary of misspelled words
        return misspelled

    except FileNotFoundError:
        # Handle the case where the file is not found
        print(f"Error: File not found: {filename}")
        return None



# spell checker function with TST
def spell_checker_tst(filename, dictTst):
    """
    Checks the spelling of words in a file against a provided dictionary TST.

    Parameters:
        filename (str): The name of the file to check.
        dictTst (Ternary Search Tree): A TST containing all valid words (dictionary).

    Returns:
        dict: A dictionary with line numbers as keys and lists of misspelled words as values.
        None: If the file is not found.
    """
    misspelled = {}  # Dictionary to store misspelled words by line number

    try:
        # Open the file in read mode with UTF-8 encoding
        with open(filename, 'r', encoding="utf-8") as file:
            line_num = 0  # To track the line number

            for line in file:
                line_num += 1  # Increment line number
                misspelled[line_num] = []  # Initialize the list for this line's misspelled words
                
                # Use regex to remove unwanted punctuation and characters
                line = re.sub(r"['’,.\"\-–;_:()!?\[\]{}]", " ", line)  # Replace common punctuation with space
                
                # Split the line into words and check each word
                for word in line.split():
                    word = word.lower()  # Normalize to lowercase
                    # If the word is not in the dictionary and it's not a number, add it to misspelled
                    if not radix_search(dictTst, word) and not check_numeric(word):
                        misspelled[line_num].append(word)

        # Return the dictionary of misspelled words
        return misspelled

    except FileNotFoundError:
        # Handle the case where the file is not found
        print(f"Error: File not found: {filename}")
        return None

def compute_words_distance(word1, word2):
    # Create a matrix to store distances
    n, m = len(word1), len(word2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]    

    # Initialize matrix with base cases
    for i in range(n + 1):
        dp[i][0] = i
    
    for j in range(m + 1):
        dp[0][j] = j
   
    # Compute distances by comparing each character of word1 
    # to all character of word2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m]

def find_top_k_similar(target_word, word_list, k=3):
    """Finds the top k most similar words based on distance."""
    if not word_list:
        return []  # Handle empty list case

    # Compute edit distances for all words
    distances = []
    for word in word_list:
        distances.append((compute_words_distance(target_word, word), word))

    # Sort manually based on distance
    for i in range(len(distances)):
        for j in range(i + 1, len(distances)):
            if distances[i][0] > distances[j][0]:  # Compare distances
                distances[i], distances[j] = distances[j], distances[i]

    # Extract the top k words
    top_k_words = [word for _, word in distances[:k]]

    return top_k_words
 
 
def spell_suggestor(misspelled, wordDict):
    """
    Corrects misspelled words by finding the top 3 similar words from a given dictionary.

    Args:
    misspelled (dict): A dictionary where values are lists of misspelled words.
    wordDict (list): A list of correctly spelled words.

    Returns:
    dict: A dictionary mapping each misspelled word to a list of the top 3 similar words.
    """

    misspelled_top3_similar = {}  # Dictionary to store corrected words

    for words in misspelled.values(): 
        for w in words:
            # Find the top 3 similar words for each misspelled word
            misspelled_top3_similar[w] = find_top_k_similar(w, wordDict, k=3)

    return misspelled_top3_similar




# Helper function to print the misspelled words
def print_misspelled(misspelled):
    """
    Helper function to print the misspelled words by line number.
    """
    for line_num, words in misspelled.items():
        if len(words) > 0:
            print(f"Line #: {line_num} - Misspelled words: {', '.join(words)}")

def print_misspelled_similar(misspelled_similar):
    """
    Helper function to print the misspelled words by line number.
    """
    for word, content in misspelled_similar.items():
        if len(content) > 0:
            print(f"Misspelled word: {word} - Spelling suggestions: {', '.join(content)}")




def main():
    
    print("\n------------START of the Program------------------\n")
    
    # Define the file paths for the dictionary and the test file
    dic_file_path = "./data/Dataset_TST/dictionary.txt"
    test_file_path = "./data/Dataset_TST/random1000words.txt"

    # Initialize TST root
    root = Node('')

    # Load the dictionary as a list
    tracemalloc.start()  # Start traccking the memory usage 

    start_time = time.time() # capture the start time
    dictList = load_text_file_to_list(dic_file_path)
    end_time = time.time() # capture the end time

    current_load_dict_list, peak_load_dict_list = tracemalloc.get_traced_memory()      
    tracemalloc.stop()

    ld_list_execution_time = (end_time - start_time)
    usage_load_dict_list=peak_load_dict_list-current_load_dict_list
     

    # Load the dictionary as  TST  
    tracemalloc.start()  # Start traccking the memory usage 

    start_time = time.time() # capture the start time 
    dictTst = load_text_file_to_tst(root, dic_file_path)
    end_time = time.time() # capture the end time

    current_load_dict_tst, peak_load_dict_tst = tracemalloc.get_traced_memory()    
    tracemalloc.stop()

    ld_tst_execution_time = (end_time - start_time)
    usage_load_dict_tst=peak_load_dict_tst-current_load_dict_tst


    #--------- Runtime Evaluation ---------------

    # Check for misspelled words using the list-based dictionary
    tracemalloc.start()  # Start traccking the memory usage
 

    start_time = time.time() # capture the start time  
    misspelled_list = spell_checker_list(test_file_path, dictList)        
    end_time = time.time() # capture the end time

    current_rt_dict_list, peak_rt_dict_list = tracemalloc.get_traced_memory()      
    tracemalloc.stop()

    rt_list_execution_time = (end_time - start_time)
    usage_rt_dict_list=peak_rt_dict_list-current_rt_dict_list


    # Check for misspelled words using the TST-based dictionary
    tracemalloc.start()  # Start traccking the memory usage
   
    start_time = time.time() # capture the start time 
    misspelled_tst = spell_checker_tst(test_file_path, dictTst)    
    end_time = time.time() # capture the end time

    current_rt_dict_tst, peak_rt_dict_tst = tracemalloc.get_traced_memory()        
    tracemalloc.stop()

    rt_tst_execution_time = (end_time - start_time)
    usage_rt_dict_tst=peak_rt_dict_tst-current_rt_dict_tst



    # -------------Print out the result -----------------------------------

    print(" ____________________________________________")
    print("|             Memory Evaluation              |")
    print("|____________________________________________|")
    print("|  Process Name            |   Memory usage  |")
    print("|____________________________________________|")
    print("|  Loading Dict in to list |",f"   {usage_load_dict_list / (1024):.4f} KB   |") 
    print("|____________________________________________|")
    print("|  Loading Dict in to TST  |",f"   {usage_load_dict_tst / (1024):.4f} KB   |") 
    print("|____________________________________________|")
    print("|  Searching word in List  |",f"   {usage_rt_dict_list / (1024):.4f} KB   |") 
    print("|____________________________________________|")
    print("|  Searching word in TST   |",f"   {usage_rt_dict_tst / (1024):.4f} KB   |") 
    print("|____________________________________________|\n")

    print(" ____________________________________________")
    print("|            Runtime Evaluation              |")
    print("|____________________________________________|")
    print("|  Process Name            |  Runtime in s   |")
    print("|____________________________________________|")
    print("|  Loading Dict into list  |",f"   {ld_list_execution_time:.2f} s       |") 
    print("|____________________________________________|")
    print("|  Loading Dict into TST   |",f"   {ld_tst_execution_time:.2f} s      |") 
    print("|____________________________________________|")
    print("|  Searching word in List  |",f"   {rt_list_execution_time:.2f} s       |") 
    print("|____________________________________________|")
    print("|  Searching word in TST   |",f"   {rt_tst_execution_time:.2f} s       |") 
    print("|____________________________________________|\n")

    print("\n--------- List Misspelled Words---------------\n")

    print("\n***Misspelled words using TST-based dictionary:")
    print_misspelled(misspelled_tst)

    if len(misspelled_tst)>0:
        suggested_words_list =spell_suggestor(misspelled_tst, dictList)
        print("\n***** Top 3 Similar word for misspelled words *****")
        print_misspelled_similar(suggested_words_list)




    print("\n------------END of the Program------------------\n")


if __name__ == "__main__":
    main()

