```bash
python -m pip install .
python -m elapsed_span --query 0 1 2 4 --reference 0 1 2 4 5 6
python -m elapsed_span --query 0 1 2 4 --reference 0 1 2 4 5 6 --search count
python -m elapsed_span --query 0 1 2 4 --reference 0 1 2 4 5 6 --search none
python -m elapsed_span --query 0 1 2 4 --reference 0 1 2 4 5 6 --family unilateral --search count
```

```python
from elapsed_span import distance, prefix_distance

query = [0.0, 1.0, 2.0, 4.0]
reference = [0.0, 1.0, 2.0, 4.0, 5.0, 6.0]

print(distance(query, reference))
print(prefix_distance(query, reference))
print(prefix_distance(query, reference, family="unilateral", search="count"))

gallery = [[0.0, 1.0, 2.0, 4.0], [0.0, 1.0, 3.0, 4.0]]
matches = [prefix_distance(query, candidate) for candidate in gallery]
ranking = sorted(range(len(gallery)), key=lambda index: matches[index].distance)
print(ranking)
```
