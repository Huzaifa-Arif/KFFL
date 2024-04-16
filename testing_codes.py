import torch

# Create a 5x5 tensor
tensor = torch.Tensor([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
])

# Calculate the overall mean of the tensor
overall_mean = torch.mean(tensor)
print("Overall Mean:", overall_mean.item())

# Calculate the mean of the tensor along the row dimension (axis=1)
row_mean = torch.mean(tensor, dim=0)
print("Row Mean:", row_mean)

# Calculate the mean of the tensor along the column dimension (axis=0)
col_mean = torch.mean(tensor, dim=1)
print("Column Mean:", col_mean)

