'''
Quiz Thursday
    - General high level ideas
    - Mixture of open response and M/C

Chapter 4 - Statistics stuff
    - Accuracy can be very misleading. Only a great solo metric for perfectly balanced data
    - Precision + Recall + F1-score 
        - Precision = Measures false positives
        - Recall = Measures false negatives
        - F-score = balance of precision and recall (beta usually == 1 --- I.E. F1-score)
    - Statistical Significance Testing
        - How do you determine if Model A is ACTUALLY better than Model B?
        - delta = (MA, x) - (MB, x) ***NON-PARAMETRIC TESTING MEASURE***
            - delta is the accuracy of model A minus the accuracy of model B
        - Use with replacement sampling of a test dataset to make thousands of test datasets and build a distribution of accuracies 
                - AKA bootstrapping (sampling with replacement)
            - IF the ACTUAL test data accuracy is rare in the virtual data it IS statistically significantly different
                - Count the number of random samples that are >= 2 * delta
                - if P value is less than the confidence interval, you reject the null hypothesis (model IS better)
                    - P value for this is % of samples in decimal form
                        - A low P value indicates outcome is likely significant (same idea; rare stuff not usually by chance)
                    - confidence is usually 0.05 (95%)
                        - The idea is that rare outcomes don't tend to happen by pure chance.
        - Works with any metric (not just accuracy)
'''

