import h5py

modality = input(">>> Input Modality You Want Here. Options are kspace and image: ")

# Open the file
with h5py.File(f"/home/Data/train/{modality}/brain_acc5_78.h5", "r") as f:
    # Print the keys (dataset names)
    print(list(f.keys()))
    dataset_name = input(">>> Input Dataset Specifics You Want Here: ")
    # Access and print details of a specific dataset
    dataset = f[dataset_name]
    print(dataset[0][0][0])
    print(dataset.shape)
    print(dataset.dtype)

    # Print attributes
    for attr in dataset.attrs:
        print(f"{attr}: {dataset.attrs[attr]}")
