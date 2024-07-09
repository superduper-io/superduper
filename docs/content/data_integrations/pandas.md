# Pandas

Although `pandas` is not a database, it came come in very handy for testing.
To connect, one specifies a list of `.csv` files:

```python
import glob
from superduper import superduper

db = superduper(glob.glob('*.csv'))
```