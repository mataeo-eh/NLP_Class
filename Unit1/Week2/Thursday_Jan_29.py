"""
N-grams...
    - Form of language models, not large, just language models.
        - LM's predict the next 'word' in the sequence
        - N-grams use Markov probability to predict the highest proba of the next word. 
            - They do not use the entire string, they just use the last word to predict the next word.
        - One wrong token early means the rest of the prediction can be very off-base with predicting one token at a time sequentially
        - N-grams can't really handle 'new' tokens - assign it zero
            - Caused some really fancy smoothing techniques to be developed. 
    - Language Model Evaluation
        - Extrinsic and Intrinsic evaluation measures
            - Intrinsic = measures the quality of a model independent of any application 
                - One such intrinisic property is perplexity
                    - Low perplexity is better
            - Extrinsic = embed it in an application and evaluate how well it does/how much it improves
                - Like LLM's an coding tasks for example
        - Both measures are important
    - Smoothing, Interpolation, and Backoff ==== VERY IMPORTANT TO REVIEW AND LEARN QUESTION MAY BE ON A QUIZ ABOUT THEM!!!!!!
"""


