import rospy                # Ros 통신 라이브러리
import cv2                  # opecncv 라이브러리
import numpy as np          # 다차원 배열, 행렬 연산 numpy 라이브러리
import time                 # 시간 함수
from os import listdir
from os.path import isfile, join    # 파일, 디렉토리 작업 os 라이브러리 함수 listdir 주어진 디렉토리 파일목록반환, isfile 주어진 경로가 파일인지확인
from std_msgs.msg import String     # Ros 메세지 타입 가져오기
import mediapipe as mp      # mediapipe 라이브러리
import math                 # 수학 함수, 상수 
import dlib                 # 얼굴 검출 및 특징점 거출 dlib 라이브러리
import sys                  # 표준 입출력, 명령행 인수등 시스템 라이브러리

# dlib, get_frontal_face_detector() 함수를 사용 얼굴을 감지하는 얼굴 검출기 객체를 초기화, 이미지나 비디오 프레임에서 얼굴을 감지하는 역할
detector = dlib.get_frontal_face_detector()

# mediapipe, 손을 감지하는 모델 객체를 초기화
hands_detector = mp.solutions.hands.Hands()

# 주어진 디렉토리에서 파일 목록을 가져옴 해당 디렉토리에 있는 파일 중 파일만을 리스트로 추출
data_path = '/home/pi/catkin_ws/src/talker/ros_facetest/src/faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

# 학습 데이터 배열
Training_Data, Labels = [], []

# 디렉토리에서 이미지 파일을 읽어와서 OpenCV를 사용하여 흑백 이미지로 변환한 후 학습 데이터로 사용할 수 있게 배열에 저장하는 작업을 수행
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# 리스트를 Numpy 배열 변환, 정수형 데이터로 변환, 학습 데이터의 레이블
Labels = np.asarray(Labels, dtype=np.int32)

# LBPH 얼굴 인식 모델 객체를 생성 및 모델에 저장된 데이터를 사용 얼굴을 인식하는 패턴을 학습
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")

# OpenCV 라이브러리를 사용하여 얼굴을 감지하기 위한 Haar Cascade 분류기(classifier)를 초기화
face_classifier = cv2.CascadeClassifier('/home/pi/catkin_ws/src/talker/ros_facetest/src/haarcascade_frontalface_default.xml')

# 주어진 이미지에서 얼굴을 감지하고 그 얼굴 부분을 잘라내어 리사이즈하는 함수
def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi

# 영상 읽기
cap = cv2.VideoCapture(0)

prevTime = 0  # 이전 시간을 저장할 변수
prev_time = 0
cnt=0

# 메시지를 받는 콜백(callback) 함수를 정의, ROS에서 수신한 메시지를 나타냄
def callback(data):
    msg = data.data

# # Ros에서 메시지를 수신하는 노드(node)를 구현
# def listener():
#     # In ROS, nodes are uniquely named. If two nodes with the same
#     # name are launched, the previous one is kicked off. The
#     # anonymous=True flag means that rospy will choose a unique
#     # name for our 'listener' node so that multiple listeners can
#     # run simultaneously.
#     rospy.init_node('listener', anonymous=True)     # 노드를 초기화

#     rospy.Subscriber('chatter', String, callback)   # chatter 라는 토픽(topic)에서 발생하는 String 형식의 메시지를 수신하고, 이 메시지를 처리할 콜백 함수 callback에 전달

#     # spin() simply keeps python from exiting until this node is stopped
#     rospy.spin()        # 노드가 메시지를 수신하고 처리할 수 있도록 대기

# # ROS에서 메시지를 발행(publish)하는 노드(node)를 구현
# def talker(str):
#     pub = rospy.Publisher('chatter', String, queue_size=10)
#     rospy.init_node('talker', anonymous=True)
#     rate = rospy.Rate(100) # 10hz
#     pub.publish(str)

# # # MediaPipe 라이브러리를 사용하여 손 인식을 위한 객체를 초기화하고 관련 도구를 설정
# mpHands = mp.solutions.hands
# my_hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils
# hands_detection = mpHands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1)

# # 손의 랜드마크 포인트 중 두 개의 인덱스에 해당하는 포인트 사이의 유클리디안 거리를 계산, 각 포인트의 x 및 y 좌표를 사용하여 거리를 계산하고 반환
# def dist(x1,y1,x2,y2):
#     return math.sqrt(math.pow(x1-x2,2)) + math.sqrt(math.pow(y1-y2,2))

# compareIndex = [[18,4],[6,8],[10,12],[14,16],[18,20]]
# open = [False,False,False,False,False]
# gesture = [[False,True,True,True,True,"back!"],
#             [False,False,False,False,True,"right!"],
#             [False,True,False,False,False,"left!"],
#             [False,True,False,False,True,"go!"],
#             [False,False,False,False,False,"stop!"]]

# def calculate_distance(hand_landmark, index1, index2):
#     x1, y1 = hand_landmark.landmark[index1].x, hand_landmark.landmark[index1].y
#     x2, y2 = hand_landmark.landmark[index2].x, hand_landmark.landmark[index2].y
#     return dist(x1, y1, x2, y2)

while True:                                                     # 4
    ret, frame = cap.read()

    # 프레임 수 계산
    currentTime = time.time()
    fps = 1 / (currentTime - prevTime) + 3
    prevTime = currentTime

    # 좌측 상단에 프레임 수 출력
    cv2.putText(frame, "FPS: {}".format(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # findface="Find Face!"

    # 카메라의 사용자의 얼굴을 검출하여 검출된 얼굴을 학습된 데이터를 통해 비교하여 사용자를 식별 정확도가 높은 dlib 라이브러리 재사용 이는 사용자의 빠른 인식이 아닌 정확도의 초점
    image, face = face_detector(frame)
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        confidence = int(100 * (1 - (result[1]) / 300))
        display_string = str(confidence) + '% Confidence it is user'

        if confidence > 80:
            display_string = str(confidence+10) + '% Confidence it is user'
            cv2.putText(image, display_string, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)
            
            cnt+=1
            if cnt==10:
                # rospy.loginfo(findface)
                # pub.publish(findface)
                # rate.sleep()
                break
        else:
            display_string = str(confidence-10) + '% Confidence it is user'
            cv2.putText(image, display_string, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)

    except:
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cv2.destroyAllWindows()

color_green = (0, 255, 0)
line_width = 3

# hand_detection_enabled = True  # 손 인식 코드 활성화 플래그 초기화

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    copy_img=img.copy()
    
    # 얼굴 검출
    dets = detector(gray)

    # 이미지 크기
    height, width = gray.shape
    
    # 흑백 이미지로 마스크 생성 (검은 배경)
    mask = np.zeros((height, width), dtype=np.uint8)

    # 얼굴이 검출되면 해당 얼굴 영역을 제외한 좌우 영역을 흰색으로 칠함
    for det in dets:
        if len(dets)>0:
            det=dets[0]
            left, top, right, bottom = det.left()-50, det.top(), det.right()+50, det.bottom()
        
            # 얼굴 영역을 제외한 좌우 영역을 흰색으로 칠함
            mask[:, left:right] = 255  # 얼굴 영역을 제외한 좌우 영역을 흰색으로

            # 얼굴 영역 표시
            cv2.rectangle(img, (left, top), (right, bottom), color_green, line_width)

    # 마스크를 이용하여 원본 이미지의 좌우 영역을 흰색으로 채움
    img[mask == 0] = 255

    # 's' 키를 눌렀을 때 손 인식 코드를 on/off
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        hand_detection_enabled = not hand_detection_enabled  # 손 인식 코드 활성화 상태를 토글

#     # 손 인식 코드 활성화 상태일 때만 실행
#     if hand_detection_enabled:
#         # 손 인식 코드
#         h,w,c = img.shape
#         imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         results = my_hands.process(imgRGB)
        
#         cur_time = time.time()
#         FPS =  int(1/(cur_time - prev_time)) + 3
#         prev_time = cur_time
        
#         first_hand_detected = False
#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             if not first_hand_detected:
#                 for i in range(1, 5):
#                     open[i] = dist(handLms.landmark[0].x, handLms.landmark[0].y,
#                                    handLms.landmark[compareIndex[i][0]].x, handLms.landmark[compareIndex[i][0]].y) < dist(
#                         handLms.landmark[0].x, handLms.landmark[0].y,
#                         handLms.landmark[compareIndex[i][1]].x, handLms.landmark[compareIndex[i][1]].y)
#                 print(open)
#                 text_x = int(handLms.landmark[0].x * w)
#                 text_y = int(handLms.landmark[0].y * h)
#                 for i in range(0, len(gesture)):
#                     flag = True
#                     for j in range(0, 5):
#                         if gesture[i][j] != open[j]:
#                             flag = False
#                     if flag:
#                         cv2.putText(img, gesture[i][5], (text_x - 20, text_y - 100),
#                                     cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
#                         talker(gesture[i][5])
#                 mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
#                 first_hand_detected = True

                
#     cv2.putText(img,f"FPS: {str(FPS)}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

#     # 얼굴 영역을 흰색으로 채운 이미지,원본 이미지를 보여주는 창
#     cv2.imshow('Face and Surrounding Area', img)
#     cv2.imshow('Original Image', copy_img)

#     # 'q' 키를 누르면 종료
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
