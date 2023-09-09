# # bottom-up cube implementation
# def buc(input,dim):
#     # input: a dataframe
#     # dim: a list of dimensions
#     # output: a dataframe
#     if len(dim) == 1:
#         print("input sum: ")
#         print(len(input))
#         return pd.DataFrame([input.sum()], columns=dim)
#     else:
#         dim0 = dim[0]
#         dim_rest = dim[1:]
#         print("dim_rest:")
#         print(len(dim_rest))
#         output = pd.DataFrame(columns=input.columns)
#         for value in input[dim0].unique():
#             sub_input = input[input[dim0] == value]
            
#             # print(value)
#             if len(sub_input) >= 1:
#                 # print("sub_input:") 
#                 # print(sub_input)
#                 sub_output = buc(sub_input, dim_rest)
#                 # print("sub_output:")
#                 # print(sub_output)
#                 sub_output[dim0] = value
#                 output = pd.concat([output, sub_output], ignore_index=True)

#         return output




def fun(cols,target):
    for i in range(len(cols)):
        if cols[i]==target:
            return i
    return -1

def buc(df,dim,row):
    #base case
    if len(dim)==0:
        #print("len 0")
        if len(df)>=2:
            row[len(row)-1]=len(df)
            #print(row)
            ans.add(tuple(row))
            row[len(row)-1]=0
        return

    first_attr=dim[0]
    remaining=dim[1:]
    # loop through the unique rows
    for attr in df[first_attr].unique():
        partial_df = df[df[first_attr]==attr]
        # print(partial_df)
        if len(partial_df)>=2:
            row[fun(df.columns,first_attr)] = attr
            row[len(row)-1]=len(partial_df)
            copy_row=list(row)
            #print(row)
            ans.add(tuple(row))
            buc(partial_df,remaining,copy_row)
            row[fun(df.columns,first_attr)] = '*'
            row[len(row)-1]=0

lst = [(i,len(df[i].unique())) for i in dim]
lst.sort(key=lambda x:x[1])
dim = [i[0] for i in lst]
row = ['*','*','*','*',0]
#buc(df,dim,row)
    