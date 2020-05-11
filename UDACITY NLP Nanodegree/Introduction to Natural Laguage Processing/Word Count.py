"""Count words.
Let's implement a simple function that is often used in Natural Language Processing: 
Counting word frequencies
"""

def count_words(text):
    import re
    """Count how many times each unique word occurs in text."""
    counts = dict()  # dictionary of { <word>: <count> } pairs to return
    
    # TODO: Convert to lowercase
    text = text.lower()
    # TODO: Split text into tokens (words), leaving out punctuation
    # (Hint: Use regex to split on non-alphanumeric characters)
    text = re.sub(r'[^A-Za-z0-9]', ' ', text)
    words = [t for t in text.split()]
    # TODO: Aggregate word counts using a dictionary
    for single_word in words:
        if single_word not in counts.keys():
            counts[single_word] = 1
        else:
            counts[single_word] += 1
            
    return counts


def test_run():
    with open("input.txt", "r") as f:
        text = f.read()
        counts = count_words(text)
        sorted_counts = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
        
        print("10 most common words:\nWord\tCount")
        for word, count in sorted_counts[:10]:
            print("{}\t{}".format(word, count))
        
        print("\n10 least common words:\nWord\tCount")
        for word, count in sorted_counts[-10:]:
            print("{}\t{}".format(word, count))


if __name__ == "__main__":
    test_run()
