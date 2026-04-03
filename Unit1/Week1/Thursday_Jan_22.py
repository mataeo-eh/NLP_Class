'''
Regex
    - The first programmatic method of searching for stuff in NLP
    - Returns the line with anything matchinhg the regex search
        - /word/
            - putting something in /***/, will return the line with whatever EXACTLY matches what is between the / /
        - /[wW]ord/
            - The [] means searching for anything within the [] - in this case word with an upper or lower W
        - /beg.n/
            - period works as a wildcard to specify any character
        - /[0-9]/
            - Searches for a range of anything between the two things connected by -
'''
"""
Herden's law (Heaps law)
    - |V| = kN^B
    - Relates the number of word types to number of instances N
        - k and beta are positive constants with 0 < beta < 1
"""
'''
Corpora
    - Over 7k languages in the world
    - Important to test algorithms across many languages
    - Code switching...
        - Multiple languages in a single communication act 
            - Talk in Spanish, a few English words, back to Spanish
'''
"""
Text Normalization
    -tokenization 
        - segmenting a sequence into words and subwords
            - top-down (rule-based) tokenization
            - byte-pair encoding (BPE - bottom up)
                - used by LLM's
                - start with subwords (arbitrary strings) token-learner progress to token segmenter
                    - token-learner = tokenization algorithm 
                    - token-segmenter = break up the input text into tokens
                - builds vocabulary of tokens. Lets models handle rare ad unseen words efficiently (the building blocks are there)
    - stop word removal
        - remove words like 'a' 'the' 'this'
            - remove very frequent and un-important words (do not give you an idea about what the document is about)
                - usually a list of words you are just going to delete from the text (somestimes call a stop word library)
"""
'''
Normalization/Lemmatization/Stemming
    - Normalization = putting text into a standard format
        - case folding
    - Lemmanization = determining if two words have the same root
        - 'He is reading detective stories' -> 'He be read detective story'
            - words like Am Are Is have the same shared lemma 'Be'
    - Stemming = simpler methods of Lemmatization
        - chop off parts of the word to approximate the root
'''
"""
Notes on Use
    - LLMs use...
        - Subword Tokenization == handles morphology
        - Contextual Embeddings == captures meaning
        - Attention == uses context dynamically
    - LLMs DO NOT USE
        - Stemming == Too lossy
        - Lemmanization == Still unnecessary
"""
'''
Sentence segmentation
    How to segment text?
    - Use cues here are punctuation markings, but there are different methods
    - Periods can be a problem...... Blah blah etc., Ph.D
'''
"""
Minimum edit distance
    - edit distances give a way to quantify intuitions about string similarity
        - What word do you think someone wanted to spell if they mis-spelled it
        - defined as the minimum number od editing operations required to turn one string into another
            - editing operations include insertion, deletion, and substitution of characters
                - minimum edit distances examples....
                    'cat' to 'cut' == MED of 1 - sub 'u' and 'a'
"""