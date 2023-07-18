def load_dataframe(sep_len,sep_wid,pet_len,pet_wid,pre_path):
    import pandas as pd
    import pickle
    xnew = pd.DataFrame([sep_len,sep_wid,pet_len,pet_wid]).T
    xnew.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    with open(pre_path,'rb') as file:
        num_pipe = pickle.load(file)

    xnew_pre = num_pipe.transform(xnew)
    xnew_pre = pd.DataFrame(xnew_pre,columns=xnew.columns)
    return xnew_pre

def load_pickle(pickle_path):
    import pickle
    with open(pickle_path,'rb') as file:
        p = pickle.load(file)
    return p