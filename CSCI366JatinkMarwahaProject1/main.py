from collections import Counter, defaultdict
import math

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.readlines()

def write_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as file:
        file.writelines(content)

def tokenize_lines(lines):
    return [token for line in lines for token in line.strip().split()]

def file_to_lowercase(filename):
    lines = read_file(filename)
    lower_lines = [line.lower() for line in lines]
    write_file(filename, lower_lines)

def add_symbols_to_lines(filename):
    lines = read_file(filename)
    modified_lines = [f"<s> {line.strip()} </s>\n" for line in lines]
    write_file(filename, modified_lines)

def add_symbols_to_lines_newfile(input_filename, output_filename):
    lines = read_file(input_filename)
    modified_lines = [f"<s> {line.strip()} </s>\n" for line in lines]
    write_file(output_filename, modified_lines)

def replace_single_occurrence_with_unk(filename, output_filename=None):
    lines = read_file(filename)
    tokens = tokenize_lines(lines)
    token_counts = Counter(tokens)
    modified_lines = [' '.join([token if token_counts[token] > 1 else '<unk>' for token in line.strip().split()]) + '\n'
                      for line in lines]
    write_file(output_filename, modified_lines)

def count_tokens_with_specials(filename):
    lines = read_file(filename)
    tokens = tokenize_lines(lines)
    token_counts = Counter(tokens)
    unique_token_count = len(token_counts) - 1
    total_token_count = sum(token_counts.values())
    unk_count = token_counts.get('<unk>', 0)
    s_count = token_counts.get('<s>', 0)
    es_count = token_counts.get('</s>', 0)
    print("\n" + "-------------------------------------------------------------")
    print(f"Total tokens: {total_token_count}")
    print(f"Unique tokens: {unique_token_count}")
    print(f"<unk> count: {unk_count}")
    print(f"<s> count: {s_count}")
    print(f"</s> count: {es_count}")

def unigram_probabilities_to_file(input_filename, output_filename):
    lines = read_file(input_filename)
    tokens = tokenize_lines(lines)
    token_counts = Counter(tokens)
    total_tokens = sum(token_counts.values())
    token_probabilities = {token: count / total_tokens for token, count in token_counts.items()}
    sorted_token_probabilities = sorted(token_probabilities.items(), key=lambda item: item[1], reverse=True)
    content = [f"{'Token':<20}{'Probability':<10}\n", "-" * 30 + "\n"]
    content += [f"{token:<20}{prob:<10.6f}\n" for token, prob in sorted_token_probabilities]
    content.append("\n" + "-" * 30 + "\n")
    content.append(f"Total tokens: {total_tokens}\n")
    write_file(output_filename, content)

def bigram_probabilities(input_filename, output_filename):
    lines = read_file(input_filename)
    tokens = tokenize_lines(lines)
    pair_counts = defaultdict(Counter)
    for i in range(len(tokens) - 1):
        pair_counts[tokens[i]][tokens[i + 1]] += 1
    content = [f"{'Word1':<15}{'Word2':<15}{'Probability':<10}\n", "-" * 40 + "\n"]
    for word1, next_words in pair_counts.items():
        total_occurrences = sum(next_words.values())
        for word2, count in next_words.items():
            conditional_prob = count / total_occurrences
            content.append(f"{word1:<15}{word2:<15}{conditional_prob:<10.6f}\n")
    write_file(output_filename, content)

def bigram_addone_probabilities(input_filename, output_filename):
    lines = read_file(input_filename)
    tokens = tokenize_lines(lines)
    token_counts = Counter(tokens)
    vocabulary_size = len(token_counts)
    pair_counts = defaultdict(Counter)
    for i in range(len(tokens) - 1):
        pair_counts[tokens[i]][tokens[i + 1]] += 1
    content = [f"{'Word1':<15}{'Word2':<15}{'Probability':<10}\n", "-" * 40 + "\n"]
    for word1, next_words in pair_counts.items():
        total_occurrences = sum(next_words.values()) + vocabulary_size
        for word2, count in next_words.items():
            if count > 0:
                smoothed_count = count + 1
                conditional_prob = smoothed_count / total_occurrences
                content.append(f"{word1:<15}{word2:<15}{conditional_prob:<10.6f}\n")
    write_file(output_filename, content)

def compare_unique_tokens(file1, file2):
    def get_tokens_and_counts(filename):
        lines = read_file(filename)
        tokens = tokenize_lines(lines)
        return set(tokens), len(tokens)
    unique_tokens_file1, total_tokens_file1 = get_tokens_and_counts(file1)
    unique_tokens_file2, total_tokens_file2 = get_tokens_and_counts(file2)
    shared_tokens = unique_tokens_file1.intersection(unique_tokens_file2)
    unique_to_file1 = unique_tokens_file1.difference(unique_tokens_file2)
    unique_to_file2 = unique_tokens_file2.difference(unique_tokens_file1)
    print('--------------------------------------')
    print(f"Unique tokens in {file1}: {len(unique_tokens_file1)}")
    print(f"Total tokens in {file1}: {total_tokens_file1}")
    print(f"Unique tokens in {file2}: {len(unique_tokens_file2)}")
    print(f"Total tokens in {file2}: {total_tokens_file2}")
    print(f"Shared tokens: {len(shared_tokens)}")
    print(f"Tokens only in {file1}: {len(unique_to_file1)}")
    print(f"Tokens only in {file2}: {len(unique_to_file2)}")

def compare_unique_bigrams(file1, file2):
    def get_bigrams_and_counts(filename):
        lines = read_file(filename)
        tokens = tokenize_lines(lines)
        bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1) if
                   tokens[i] != '<s>' and tokens[i + 1] != '<s>']
        return Counter(bigrams), set(bigrams)
    bigrams_file1, unique_bigrams_file1 = get_bigrams_and_counts(file1)
    bigrams_file2, unique_bigrams_file2 = get_bigrams_and_counts(file2)
    shared_bigrams = unique_bigrams_file1.intersection(unique_bigrams_file2)
    unique_to_file1 = unique_bigrams_file1.difference(unique_bigrams_file2)
    unique_to_file2 = unique_bigrams_file2.difference(unique_bigrams_file1)
    total_shared_occurrences_file1 = sum(bigrams_file1[bigram] for bigram in shared_bigrams)
    total_shared_occurrences_file2 = sum(bigrams_file2[bigram] for bigram in shared_bigrams)
    print(f"Unique bigrams in {file1}: {len(unique_bigrams_file1)}")
    print(f"Total bigrams in {file1}: {sum(bigrams_file1.values())}")
    print(f"Unique bigrams in {file2}: {len(unique_bigrams_file2)}")
    print(f"Total bigrams in {file2}: {sum(bigrams_file2.values())}")
    print(f"Shared bigrams: {len(shared_bigrams)}")
    print(f"Total occurrences of shared bigrams in {file1}: {total_shared_occurrences_file1}")
    print(f"Total occurrences of shared bigrams in {file2}: {total_shared_occurrences_file2}")
    print(f"Bigrams only in {file1}: {len(unique_to_file1)}")
    print(f"Bigrams only in {file2}: {len(unique_to_file2)}")

def get_text_log_probability(text, unigram_probs):
    words = text.strip().split()
    total_log_probability = 0.0
    total_words = 0
    print(f"{'Word':<15}{'Probability':<15}{'Log Probability':<20}{'Running Total Log':<20}")
    print("-" * 70)
    for word in words:
        word = word.strip()
        if word in unigram_probs:
            prob = unigram_probs[word]
        else:
            prob = 0.0
        if prob > 0:
            log_prob = math.log2(prob)
        else:
            log_prob = float('-inf')
        total_log_probability += log_prob
        total_words += 1
        print(f"{word:<15}{prob:<15.10f}{log_prob:<20.10f}{total_log_probability:<20.10f}")
        if log_prob == float('-inf'):
            break
    return total_log_probability, total_words

def calculate_perplexity(total_log_probability, total_words):
    if total_log_probability == float('-inf'):
        return float('inf')
    else:
        return 2 ** (-total_log_probability / total_words)

def replace_unknowns_with_unk(text, unigram_probs):
    words = text.strip().split()
    modified_words = []
    for word in words:
        if word in unigram_probs:
            modified_words.append(word)
        else:
            modified_words.append("<unk>")
    return ' '.join(modified_words)

def process_text_with_unk_replacement(text_file, unigram_file, output_file):
    unigram_probs = load_unigram_probabilities(unigram_file)
    document_text = ''.join(read_file(text_file))
    modified_text = replace_unknowns_with_unk(document_text, unigram_probs)
    write_file(output_file, [modified_text])
    print(f"Modified text has been saved to {output_file}")

def calculate_document_unigram_probability(text_file, unigram_file, output_file):
    unigram_probs = load_unigram_probabilities(unigram_file)
    document_text = ''.join(read_file(text_file))
    total_log_probability, total_words = get_text_log_probability(document_text, unigram_probs)
    perplexity = calculate_perplexity(total_log_probability, total_words)
    content = [f"{'Word':<20}{'Probability':<20}{'Log Probability':<20}{'Running Total Log':<20}\n"]
    content.append("-" * 80 + "\n")
    for word in document_text.strip().split():
        prob = unigram_probs.get(word, 0.0)
        log_prob = math.log2(prob) if prob > 0 else float('-inf')
        content.append(f"{word:<20}{prob:<20.10f}{log_prob:<20.10f}{total_log_probability:<20.10f}\n")
    content.append(f"\nTotal Log Probability (base 2): {total_log_probability}\n")
    content.append(f"Perplexity: {perplexity}\n")
    write_file(output_file, content)

def calculate_document_bigram_probability(text_file, bigram_file, output_file):
    bigram_probs = load_bigram_probabilities(bigram_file)
    lines = read_file(text_file)
    content = []
    for line in lines:
        bigram_content, total_log_probability, valid_bigrams = get_sentence_bigram_probability(line, bigram_probs)
        perplexity = calculate_perplexity(total_log_probability, valid_bigrams)
        content.append(f"Sentence: {line.strip()}\n")
        content.extend(bigram_content)  # Add detailed bigram content
        content.append(f"Total Log Probability (base 2): {total_log_probability}\n")
        content.append(f"Perplexity: {perplexity}\n")
        content.append("\n" + "=" * 75 + "\n")
    write_file(output_file, content)
def calculate_document_bigram_probability_with_smoothing(text_file, bigram_file, unigram_file, document_file, output_file):
    bigram_probs = load_bigram_probabilities(bigram_file)
    unigram_counts = load_unigram_counts(unigram_file)
    total_word_tokens = count_total_word_tokens(document_file)
    vocab_size = len(unigram_counts)
    smoothing = 1.0
    total_log_probability_document = 0.0
    total_bigrams_document = 0
    lines = read_file(text_file)
    content = []
    for line in lines:
        words = line.strip().split()
        if len(words) < 2:
            continue
        for i in range(len(words) - 1):
            word1 = words[i]
            word2 = words[i + 1]
            bigram = f"{word1} {word2}"
            if bigram in bigram_probs:
                bigram_prob = bigram_probs[bigram]
            else:
                word1_count = unigram_counts.get(word1, 0)
                bigram_prob = (word1_count + smoothing) / (total_word_tokens +  vocab_size)
            log_prob = calculate_log_probability(bigram_prob)
            total_log_probability_document += log_prob
            total_bigrams_document += 1
            content.append(f"{bigram}: log_prob = {log_prob:.6f}\n")
    if total_bigrams_document > 0:
        overall_perplexity = calculate_perplexity(total_log_probability_document, total_bigrams_document)
        content.append(f"\nTotal Log Probability (base 2) for the entire document: {total_log_probability_document}\n")
        content.append(f"Perplexity for the entire document: {overall_perplexity}\n")
    else:
        content.append("No bigrams found in the document.\n")
    write_file(output_file, content)

def get_sentence_bigram_probability(sentence, bigram_probs):
    words = sentence.strip().split()
    total_log_probability = 0.0
    valid_bigrams = 0
    content = [f"{'Bigram':<20}{'Probability':<15}{'Log Probability':<20}{'Running Total Log':<20}\n"]
    content.append("-" * 75 + "\n")
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i + 1]}"
        if bigram in bigram_probs:
            prob = bigram_probs[bigram]
        else:
            prob = 0.0
        if prob > 0:
            log_prob = math.log2(prob)
        else:
            log_prob = float('-inf')
        if log_prob == float('-inf'):
            total_log_probability = float('-inf')
        if total_log_probability != float('-inf'):
            total_log_probability += log_prob
            valid_bigrams += 1
        content.append(f"{bigram:<20}{prob:<15.10f}{log_prob:<20.10f}{total_log_probability:<20.10f}\n")
    return content, total_log_probability, valid_bigrams

def load_unigram_probabilities(unigram_filename):
    unigram_probs = {}
    lines = read_file(unigram_filename)
    for line in lines:
        if line.strip() == "" or "Token" in line:
            continue
        parts = line.strip().split()
        if len(parts) == 2:
            word, prob = parts
            word = word.strip()
            try:
                unigram_probs[word] = float(prob)
            except ValueError:
                print(f"Skipping line due to invalid probability: {line.strip()}")
                continue
    return unigram_probs

def load_bigram_probabilities(bigram_filename):
    bigram_probs = {}
    lines = read_file(bigram_filename)
    for line in lines:
        if line.strip() == "" or "Word1" in line or "-" in line:
            continue
        parts = line.strip().split()
        if len(parts) == 3:
            word1, word2, prob = parts
            bigram = f"{word1} {word2}"
            bigram_probs[bigram] = float(prob)
    return bigram_probs

def load_unigram_counts(unigram_filename):
    unigram_counts = Counter()
    lines = read_file(unigram_filename)
    for line in lines:
        if line.strip() == "" or "Word" in line or "-" in line:
            continue
        parts = line.strip().split()
        if len(parts) == 2:
            word, count = parts
            try:
                unigram_counts[word] = int(count)
            except ValueError:
                continue
    return unigram_counts

def count_total_word_tokens(document_filename):
    total_words = 0
    lines = read_file(document_filename)
    for line in lines:
        words = line.strip().split()
        total_words += len(words)
    return total_words

def calculate_log_probability(probability):
    import math
    return math.log2(probability)

file_to_lowercase('train-Fall2024.txt')
replace_single_occurrence_with_unk('train-Fall2024.txt', 'train-Fall2024-unk.txt')
add_symbols_to_lines('train-Fall2024-unk.txt')
count_tokens_with_specials('train-Fall2024-unk.txt')
unigram_probabilities_to_file('train-Fall2024-unk.txt', 'unigram_probabilities.txt')
bigram_probabilities('train-Fall2024-unk.txt', 'bigram_probabilities.txt')
bigram_addone_probabilities('train-Fall2024-unk.txt', 'bigram_addone_probabilities.txt')
add_symbols_to_lines_newfile('train-Fall2024.txt', 'train-Fall2024-nounk.txt')
file_to_lowercase('test.txt')
add_symbols_to_lines_newfile('test.txt', 'test-nounk.txt')
compare_unique_tokens('test-nounk.txt', 'train-Fall2024-nounk.txt')
process_text_with_unk_replacement('test-nounk.txt', 'unigram_probabilities.txt', 'test-unk-replacement.txt')
bigram_probabilities('test-unk-replacement.txt', 'test-bigram.txt')
compare_unique_bigrams('test-bigram.txt', 'bigram_probabilities.txt')
file_to_lowercase('1-3-5.txt')
add_symbols_to_lines('1-3-5.txt')

calculate_document_unigram_probability('1-3-5.txt', 'unigram_probabilities.txt', '1-3-5-unigram.txt')
calculate_document_bigram_probability('1-3-5.txt', 'bigram_probabilities.txt', '1-3-5-bigram.txt')
calculate_document_bigram_probability_with_smoothing('1-3-5.txt', 'bigram_addone_probabilities.txt', 'unigram_probabilities.txt', 'train-Fall2024-unk.txt', '1-3-5-bigramaddone.txt')
calculate_document_unigram_probability('test-unk-replacement.txt', 'unigram_probabilities.txt','test-unk-replacement-unigram.txt')
calculate_document_bigram_probability('test-unk-replacement.txt', 'bigram_probabilities.txt','test-unk-replacement-bigram.txt')
calculate_document_bigram_probability_with_smoothing('test-unk-replacement.txt', 'bigram_addone_probabilities.txt', 'unigram_probabilities.txt', 'train-Fall2024-unk.txt','test-unk-replacement-bigramaddone.txt')
