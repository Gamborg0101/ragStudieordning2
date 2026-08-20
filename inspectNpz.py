from numpy import load

data = load("data/vector_store.npz", allow_pickle=True)

lst = data.files
for item in lst:
    # print(item)
    print(data[item])
