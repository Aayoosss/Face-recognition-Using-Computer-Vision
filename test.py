import os
from pipeline.recognise import recognise
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


test_path = "Faces/test_images"
testimages = os.listdir(test_path)
testimages = [x for x in testimages if x not in [".DS_Store", "__MACOSX"]]
y_true = [x.split('_')[0] for x in testimages if x not in [".DS_Store", "__MACOSX"]]
# print(testimages)

y_pred = []
y_sim = []
for image in testimages:
    image_path = os.path.join(test_path, image)
    print(f"Recognising image: {image_path}")
    person, max_similarity = recognise(img_path = image_path)
    y_pred.append(person)
    y_sim.append(max_similarity)
    print("-------------------------------------------------------------------------------")
    # print(f"Detected: {person} - {max_similarity}")

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average = 'weighted')
print(f"accuracy - {accuracy}")
print(f"F1_score - {f1}")
conf = confusion_matrix(y_true, y_pred)
print(conf)

