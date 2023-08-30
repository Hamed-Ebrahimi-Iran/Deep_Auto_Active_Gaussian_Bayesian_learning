

---

- **How to train**

  1. download or extract the features.
  2. use *make_list.py* in the *list* folder to generate the training and test list.
  3. change the parameters in option.py 
  4. run *main.py*

- **How to test**

  run *infer.py* and the model is in the ckpt folder.

---


---
**In order to make training process faster, we suggest use the following code to replace original code in train.py [Line 34]**
```python
model.train()
n_iter = iter(nloader)
a_iter = iter(aloader)
for i in range(30):  # 800/batch_size
    ninput = next(n_iter)
    ainput = next(a_iter)
```

Thanks for your attention!
# DeepActiveMIL

