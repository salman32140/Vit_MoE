import os
import re

def generate_mapping(path):
    class_name_mapping = {}
    class_mapping_2 = []
    for index, dirname in enumerate(os.listdir(path)):
        if os.path.isdir(os.path.join(path, dirname)):
            class_name_mapping[dirname] = index
            class_mapping_2.append(dirname)
    with open('class_name_mapping.py', 'w') as f:
        f.write('class_name_mapping = ' + str(class_name_mapping) + '\n\n')
        f.write('class_names = ' + str(class_mapping_2) + '\n\n')
        f.write('def get_labels(class_name):\n')
        f.write('\tone_hot = np.zeros(len(class_names))\n')
        for class_name, index in class_name_mapping.items():
            f.write('\tone_hot[{}] = 1 if class_name == "{}" else 0\n'.format(index, class_name))
        
        f.write('\n\treturn torch.tensor(one_hot, dtype=torch.float32)')

# Call the function with the path to the directory
target_dataset_url = '../../../dataset/plantVillage_PRUNED_200'


# rename_folders(target_dataset_url)
generate_mapping(target_dataset_url)
