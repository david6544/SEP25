

### 1. **1-D, 10 elements, 7 queries**

```bash
seq 0.1 0.1 0.7 | ./build/sep25_main_dev 1 10 7
```


### 2. **1-D, 100 elements, 70 queries**

```bash
seq 0.1 0.1 7.0 | ./build/sep25_main_dev 1 100 70
```


### 3. **1-D, 100 elements, 10 queries**

```bash
seq 0.1 0.1 1.0 | ./build/sep25_main_dev 1 100 10
```


### 4. **2-D, 10 elements, 70 queries**

```bash
seq 0.1 0.1 7.0 | ./build/sep25_main_dev 2 10 70
```

⚡ Tip: `seq start step end` generates floats easily.
So `seq 0.1 0.1 7.0` → `0.1, 0.2, 0.3 … 7.0` (exactly 70 numbers).