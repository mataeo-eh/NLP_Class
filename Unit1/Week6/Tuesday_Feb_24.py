'''
- T-SNE
    - Known for taking high dimensional data and projecting it into 2d - 3d space
    - differences in high dimensions represented in low dimensions
        - things close to eachother in low dimensions also close in high dimensions
            - good data visualization tool
- U-MAP
    - Another clustering mechanism
        - Uncovers larger structure within the dataset you have
        - Finds the "big segments"
- Cosine similarity
    - Measuring how similar two vectors are 
    - 1.0 = identical direction
    - 0.0 = unrelated
    - -1.0 = exactly opposite directions 
- TF-IDF
    - Term Frequency - Inverse Document Frequency
        - Corrects for document length and how often a term appears in it
        - Finds words common across all of the documents, and makes them extremely low weight (even to 0)
            - Can't discriminate differences in documents if every type of document has this word :shrug:
- Pointwise Mutual Information (PMI)
    - Measure of how often two events occur, compared with what we would expect if they were independent
        - if PMI = 1, the words are not related at all, they appear totally randomly
        - Measure of if words are related basically
            - smaller non-zero and positive number = more likely to appear together 
'''

