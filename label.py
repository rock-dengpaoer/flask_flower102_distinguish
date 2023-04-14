import scipy.io as scio

def main():

    with open("Oxford-102_Flower_dataset_labels.txt") as f:
        labels = f.readlines()
    f.close()
    print(labels)
    label_list = []
    for label in labels:
        label_list.append(label.split("\n")[0].split("'")[1])
        # print(label)
    print(label_list)

    print(label_list[0])

if __name__ == "__main__":
    main()