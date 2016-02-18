# The simplest implementation of the Extreme Learning Machine algorithm

The Extreme Learning Machine (ELM) is a Single Layer FeedForward Neural Network designed by Huang et Al [1]

```python
from ELM import ELMRegressor

elm = ELMRegressor(n_hidden_units=100)
elm.fit(train_x, train_y)

prediction = elm.predict(test_x)
```

You can see a more detailed Example of how to use it on example.py



## References

[1] G.-B. Huang, Q.-Y. Zhu and C.-K. Siew, "Extreme Learning Machine:
          Theory and Applications", Neurocomputing, vol. 70, pp. 489-501,
          2006.
