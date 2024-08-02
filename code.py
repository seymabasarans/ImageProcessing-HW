import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

path = "/Users/seymabasaran/PycharmProjects/DIPHW3/color-1/valid"

classes = ["red", "blue", "green", "gray", "white"]

def selected_img(path, classes, num_images=20):
    color_array = []
    for class_name in classes:
        class_images = []
        for image_name in os.listdir(path):
            if class_name in image_name:
                image_path = os.path.join(path, image_name)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                class_images.append(image)
                if len(class_images) == num_images: #bir renkten 20 resim bulunca çık
                    break
        color_array.extend(class_images) #listeye class_image listesinin ögelerini ekledik
    return color_array

def calculate_rgb_hist(img):
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])

    #[0]görüntünün yüksekliği(height) [1] görüntünün genişliği. m*n ptoplam piksel bulundu
    pixels = img.shape[0] * img.shape[1]
    hist_b_normalize = hist_b / pixels
    hist_g_normalize = hist_g / pixels
    hist_r_normalize = hist_r / pixels

    return hist_b_normalize, hist_g_normalize, hist_r_normalize

def hist_for_RGB(color_array):
    all_histograms = []
    for image in color_array:
        hist_blue, hist_green, hist_red = calculate_rgb_hist(image)
        all_histograms.append((hist_blue, hist_green, hist_red))
    return all_histograms

def calculate_distance(hist1, hist2):
    diff = np.array(hist1) - np.array(hist2)
    squared_diff = diff**2
    sum_squared = np.sum(squared_diff)
    euclidean_dist = np.sqrt(sum_squared)
    return euclidean_dist

def k_means(all_histograms, k=5):
    last_accuracy = 0
    centers = random.sample(all_histograms, k)
    for _ in range(10):

        for _ in range(10):
            dist = [calculate_distance(hist, center) for hist in all_histograms for center in centers]
            cluster_labels = np.argmin(np.array(dist).reshape(len(all_histograms), k), axis=1)

            new_centers = [np.mean(np.array(all_histograms)[cluster_labels == i], axis=0) for i in range(k)]
            centers = new_centers

            true_labels = np.repeat(np.arange(5), 20)
            accuracy = accuracy_score(true_labels, cluster_labels)

            if accuracy > last_accuracy:
                last_accuracy = accuracy

        return cluster_labels, true_labels, last_accuracy


def get_img(color_array, true_labels, cluster_labels):
    plt.figure(figsize=(15, 8))

    for i in range(5):
        correct = np.where((true_labels == i) & (cluster_labels == i))[0]
        for j in range(min(5, len(correct))):
            plt.subplot(5, 10, i * 10 + j + 1)
            plt.imshow(color_array[correct[j]])
            plt.axis("off")
            plt.title(f"Gerçek: {i}\nCluster: {i}")

        wrong = np.where((true_labels == i) & (cluster_labels != i))[0]
        for j in range(min(1, len(wrong))):
            plt.subplot(5, 10, i * 10 + 5 + j + 1)
            plt.imshow(color_array[wrong[j]])
            plt.axis("off")
            plt.title(f"Gerçek: {i}\nCluster: {cluster_labels[wrong[j]]}")

    plt.tight_layout()
    plt.show()



def main():
    color_array = selected_img(path, classes)
    histograms = hist_for_RGB(color_array)
    cluster_labels, true_labels, accuracy = k_means(histograms)
    confusion_mat = confusion_matrix(true_labels, cluster_labels)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Karmaşıklık Matrisi: \n {confusion_mat}")
    get_img(color_array, true_labels, cluster_labels)


if __name__ == "__main__":
    main()






