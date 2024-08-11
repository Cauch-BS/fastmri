import numpy as np

result_dir = "../result/test_FM"
validation_loss = np.load(result_dir + "/val_loss_log.npy")

for i in range(len(validation_loss)):
    print(f"For Epoch {i + 1}, Validation Loss was {1 - validation_loss[i][1]/51}")
