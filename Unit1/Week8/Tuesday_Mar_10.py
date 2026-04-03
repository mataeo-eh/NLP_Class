'''
Recurrent Neural Networks
    - RNN's have _some_ memory
        - The output of the hidden layer feeds into the input of itself
            - When new information comes in, the hidden layer already has some information about what came before it
                - New input is influenced by the previous input/output
    - Classic problems
        - Vanishing/exploding gradient
            - Vanishing gradient = eventually just stops getting any meaningful update (small numbers * small numbers basically to 0)
                - LSTM and Gradient Recurrent Unit are the fixes to address vanishing gradients 
            - Exploding gradients = Eventually gradient goes opposite way and explodes
                - Gradient clipping
        - Problem with long term concepts/patterns/context
            - Related to the vanishing gradients problem. When the 'memory' gets too long, it just loses all the context from earlier.
LSTM on Thursday 
Transformers and attention after the break 

'''