Hello! this is a readme for my project
first i would like to discuss the functions and there use caes
lastly i will give a brief overview of all the output files
List of functons:
read_file(filename) : helper function to read files
write_file(filename, content) : helper function to write new files
tokenize_lines(lines) : looks for the tokens
file_to_lowercase(filename) : makes any uppercase to lowercase in the file
add_symbols_to_lines(filename) : adds the <s> and </s>
add_symbols_to_lines_newfile(input_filename, output_filename) : adds the <s> and </s> only to a new output file
replace_single_occurrence_with_unk(filename, output_filename) : replaces all tokens which appear once with <unk>
count_tokens_with_specials(filename): it counts all the tokens : including the <s> and </s> separate and the amount of unique tokens
unigram_probabilities_to_file(input_filename, output_filename) : it calculates the unigram_probabilities of a input and outputs it as a reference chart in the output
bigram_probabilities(input_filename, output_filename) : it calculates the bigram probabilities of a input and outputs it as a reference chart in the output
bigram_addone_probabilities(input_filename, output_filename): it calculates the bigram plus 1 probabilities of a input and outputs it as a reference chart in the output
compare_unique_tokens(file1, file2) : compares the unique tokens between two files
compare_unique_bigrams(file1, file2) : compares the unique bigrams between two files
get_text_log_probability(text, unigram_probs) : searches the reference table for the unigrams present
calculate_perplexity(total_log_probability, total_words) : calculates perplexity or the log probabilities over the number of tokens
replace_unknowns_with_unk(text, unigram_probs) : used to replace words in the input text with unk based on if it shows up in the unigram reference table
process_text_with_unk_replacement(text_file, unigram_file, output_file): works in conjunction with the previous function to input and output
calculate_document_unigram_probability(text_file, unigram_file, output_file) : calculate the unigram probabilities for a entire document
calculate_document_bigram_probability(text_file, bigram_file, output_file) : calculate the unigram bigram probabilities :
calculate_document_bigram_probability_with_smoothing(text_file, bigram_file, unigram_file, document_file, output_file)
get_sentence_bigram_probability(sentence, bigram_probs): calcultes the bigram probability based on a sentence
load_unigram_probabilities(unigram_filename): loads the unigram probabilites to assist calculations
load_bigram_probabilities(bigram_filename) : loads the bigram_probabilites to asset calculation
load_unigram_counts(unigram_filename) : loads the count of unigrams for the vocab
count_total_word_tokens(document_filename) : provides the total_word_tokens for smoothing
calculate_log_probability(probability): calculates the log base 2 probability

input files
1-3-5.txt : just the sentence from 1-3-5
test.txt : the test file
train-Fall2024.txt


output files
1-3-5-bigram.txt- the bigram probability and perplexity for 1-3-5
1-3-5-bigramaddone.txt - the bigram+1 probability and perplexity  for 1-3-5
1-3-5-unigram.txt - the unigram probability and perplexity for 1-3-5
bigram_addone_probabilities : its the bigram+1 probability reference table based on the cleaned training data
bigram_probables: its the bigram probability reference table based on the cleaned training data
test-bigram.txt : its the bigram probability reference table based on the cleaned test data barring the adding of unknowns based on the cleaned training data
test-nounk.txt : its the test after cleaning by lowercasing and adding the <s> and </s> token
test-unk-replacement.txt : its the test after replacing the words that dont appear in training with unk
test-unk-replacement-bigram.txt :the bigram probability and perplexity for test
test-unk-replacement-bigramaddone.txt : the bigramaddone probability and perplexity for test
test-unk-replacement-unigram.txt :the unigram probability and perpelixty for test
train_fall204-nounk: training coropus but with symbols and lowercased
train-fall2024-unk.txt: training corpus but with unk
unigram_Probabilites: the unigram probability refrence table for the training data



how to run it

just hit the run button and it all should do its thing . 
in the console at the top will be stuff to answer question 1 and 2 
all those output files should be created 