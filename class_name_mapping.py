class_name_mapping = {'tomato_septoria_leaf_spot': 0, 'apple_cedar_apple_rust': 1, 'cherry_healthy': 2, 'corn_common_rust': 3, 'tomato_leaf_mold': 4, 'apple_apple_scab': 5, 'strawberry_leaf_scorch': 6, 'tomato_spider_mites_two_spotted_spider_mite': 7, 'pepper_bell_bacterial_spot': 8, 'potato_late_blight': 9, 'peach_healthy': 10, 'tomato_target_spot': 11, 'grape_leaf_blight': 12, 'raspberry_healthy': 13, 'peach_bacterial_spot': 14, 'potato_early_blight': 15, 'pepper_bell_healthy': 16, 'corn_cercospora_leaf_spot_gray_leaf_spot': 17, 'corn_healthy': 18, 'orange_haunglongbing': 19, 'grape_healthy': 20, 'corn_northern_leaf_blight': 21, 'apple_healthy': 22, 'apple_black_rot': 23, 'tomato_tomato_mosaic_virus': 24, 'blueberry_healthy': 25, 'potato_healthy': 26, 'cherry_powdery_mildew': 27, 'tomato_bacterial_spot': 28, 'tomato_late_blight': 29, 'tomato_tomato_yellow_leaf_curl_virus': 30, 'tomato_healthy': 31, 'soybean_healthy': 32, 'strawberry_healthy': 33, 'tomato_early_blight': 34, 'squash_powdery_mildew': 35, 'grape_black_rot': 36, 'grape_esca': 37}

class_names = ['tomato_septoria_leaf_spot', 'apple_cedar_apple_rust', 'cherry_healthy', 'corn_common_rust', 'tomato_leaf_mold', 'apple_apple_scab', 'strawberry_leaf_scorch', 'tomato_spider_mites_two_spotted_spider_mite', 'pepper_bell_bacterial_spot', 'potato_late_blight', 'peach_healthy', 'tomato_target_spot', 'grape_leaf_blight', 'raspberry_healthy', 'peach_bacterial_spot', 'potato_early_blight', 'pepper_bell_healthy', 'corn_cercospora_leaf_spot_gray_leaf_spot', 'corn_healthy', 'orange_haunglongbing', 'grape_healthy', 'corn_northern_leaf_blight', 'apple_healthy', 'apple_black_rot', 'tomato_tomato_mosaic_virus', 'blueberry_healthy', 'potato_healthy', 'cherry_powdery_mildew', 'tomato_bacterial_spot', 'tomato_late_blight', 'tomato_tomato_yellow_leaf_curl_virus', 'tomato_healthy', 'soybean_healthy', 'strawberry_healthy', 'tomato_early_blight', 'squash_powdery_mildew', 'grape_black_rot', 'grape_esca']

def get_labels(class_name):
	one_hot = np.zeros(len(class_names))
	one_hot[0] = 1 if class_name == "tomato_septoria_leaf_spot" else 0
	one_hot[1] = 1 if class_name == "apple_cedar_apple_rust" else 0
	one_hot[2] = 1 if class_name == "cherry_healthy" else 0
	one_hot[3] = 1 if class_name == "corn_common_rust" else 0
	one_hot[4] = 1 if class_name == "tomato_leaf_mold" else 0
	one_hot[5] = 1 if class_name == "apple_apple_scab" else 0
	one_hot[6] = 1 if class_name == "strawberry_leaf_scorch" else 0
	one_hot[7] = 1 if class_name == "tomato_spider_mites_two_spotted_spider_mite" else 0
	one_hot[8] = 1 if class_name == "pepper_bell_bacterial_spot" else 0
	one_hot[9] = 1 if class_name == "potato_late_blight" else 0
	one_hot[10] = 1 if class_name == "peach_healthy" else 0
	one_hot[11] = 1 if class_name == "tomato_target_spot" else 0
	one_hot[12] = 1 if class_name == "grape_leaf_blight" else 0
	one_hot[13] = 1 if class_name == "raspberry_healthy" else 0
	one_hot[14] = 1 if class_name == "peach_bacterial_spot" else 0
	one_hot[15] = 1 if class_name == "potato_early_blight" else 0
	one_hot[16] = 1 if class_name == "pepper_bell_healthy" else 0
	one_hot[17] = 1 if class_name == "corn_cercospora_leaf_spot_gray_leaf_spot" else 0
	one_hot[18] = 1 if class_name == "corn_healthy" else 0
	one_hot[19] = 1 if class_name == "orange_haunglongbing" else 0
	one_hot[20] = 1 if class_name == "grape_healthy" else 0
	one_hot[21] = 1 if class_name == "corn_northern_leaf_blight" else 0
	one_hot[22] = 1 if class_name == "apple_healthy" else 0
	one_hot[23] = 1 if class_name == "apple_black_rot" else 0
	one_hot[24] = 1 if class_name == "tomato_tomato_mosaic_virus" else 0
	one_hot[25] = 1 if class_name == "blueberry_healthy" else 0
	one_hot[26] = 1 if class_name == "potato_healthy" else 0
	one_hot[27] = 1 if class_name == "cherry_powdery_mildew" else 0
	one_hot[28] = 1 if class_name == "tomato_bacterial_spot" else 0
	one_hot[29] = 1 if class_name == "tomato_late_blight" else 0
	one_hot[30] = 1 if class_name == "tomato_tomato_yellow_leaf_curl_virus" else 0
	one_hot[31] = 1 if class_name == "tomato_healthy" else 0
	one_hot[32] = 1 if class_name == "soybean_healthy" else 0
	one_hot[33] = 1 if class_name == "strawberry_healthy" else 0
	one_hot[34] = 1 if class_name == "tomato_early_blight" else 0
	one_hot[35] = 1 if class_name == "squash_powdery_mildew" else 0
	one_hot[36] = 1 if class_name == "grape_black_rot" else 0
	one_hot[37] = 1 if class_name == "grape_esca" else 0

	return torch.tensor(one_hot, dtype=torch.float32)