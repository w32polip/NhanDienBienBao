import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from keras.preprocessing import image

#############################################
#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)

threshold = 0.75 #THRESHOLD của Xác Suất
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, 640) # Chiều rộng cửa sổ
cap.set(4, 480) # Chiều dài cửa sổ
cap.set(10, 180) # Độ sáng
# IMPORT TRAINED MODEL
model = load_model('model.h5')

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def getCalssName(classNo):
    if classNo == 0:
        return 'Tốc độ tối đa 20 km/h'
    elif classNo == 1:
        return 'Tốc độ tối đa 30 km/h'
    elif classNo == 2:
        return 'Tốc độ tối đa 50 km/h'
    elif classNo == 3:
        return 'Tốc độ tối đa 60 km/h'
    elif classNo == 4:
        return 'Tốc độ tối đa 70 km/h'
    elif classNo == 5:
        return 'Tốc độ tối đa 80 km/h'
    elif classNo == 6:
        return 'End of Speed Limit 80 km/h'
    elif classNo == 7:
        return 'Tốc độ tối đa 100 km/h'
    elif classNo == 8:
        return 'Tốc độ tối đa 120 km/h'
    elif classNo == 9:
        return 'Không vượt'
    elif classNo == 10:
        return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11:
        return 'Right-of-way at the next intersection'
    elif classNo == 12:
        return 'Priority road'
    elif classNo == 13:
        return 'Đường ưu tiên'
    elif classNo == 14:
        return 'Dừng lại'
    elif classNo == 15:
        return 'No vechiles'
    elif classNo == 16:
        return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17:
        return 'Không quay đầu'
    elif classNo == 18:
        return 'Cẩn thật'
    elif classNo == 19:
        return 'Chỗ ngoặt nguy hiểm vòng bên trái'
    elif classNo == 20:
        return 'Chỗ ngoặt nguy hiểm vòng bên phải'
    elif classNo == 21:
        return 'Double curve'
    elif classNo == 22:
        return 'Đường gập ghềnh'
    elif classNo == 23:
        return 'Đường trơn trượt'
    elif classNo == 24:
        return 'Road narrows on the right'
    elif classNo == 25:
        return 'Đường đang thi công'
    elif classNo == 26:
        return 'Traffic signals'
    elif classNo == 27:
        return 'Pedestrians'
    elif classNo == 28:
        return 'Children crossing'
    elif classNo == 29:
        return 'Bicycles crossing'
    elif classNo == 30:
        return 'Beware of ice/snow'
    elif classNo == 31:
        return 'Wild animals crossing'
    elif classNo == 32:
        return 'End of all speed and passing limits'
    elif classNo == 33:
        return 'Turn right ahead'
    elif classNo == 34:
        return 'Turn left ahead'
    elif classNo == 35:
        return 'Ahead only'
    elif classNo == 36:
        return 'Go straight or right'
    elif classNo == 37:
        return 'Go straight or left'
    elif classNo == 38:
        return 'Keep right'
    elif classNo == 39:
        return 'Keep left'
    elif classNo == 40:
        return 'Roundabout mandatory'
    elif classNo == 41:
        return 'End of no passing'
    elif classNo == 42:
        return 'End of no passing by vechiles over 3.5 metric tons'

while True:
    # Đọc ảnh từ Webcame
    success, imgOrignal = cap.read()

    # Xử lý ảnh
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "Biển: ", (20, 35), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    # Tiến hành dự đoán kết quả
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
        print(getCalssName(classIndex))
        cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35),
                font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + " %", (180, 75),
                font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Demo", imgOrignal)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break