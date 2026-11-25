Mini-batches have 10 samples, and training uses SGD, with learning rate of 0.1, no momentum and no weight decay.

We consider replay buffers of capacity b ∈{0,200,500,1000,2000,5000,10000,50000}, where 50000 corresponds to the size of the entire training data, with a replay mini-batch of 10. 

The second factor we consider is the number updates per mini-batch u ∈{1,3,5,50}, where 50 is taken to approach an offline setting.

All experiments are repeated with three random seeds for initialization. We instantiate the linear probe as a linear classifier fitted with SGD per task on standardized training features and evaluated on the test set. 

NC metrics are computed on training features, and averaged over task heads.
