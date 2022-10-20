import cv2
import numpy as np
import face_recognition
import os
import sys, getopt

path = 'trainning'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)


def main(argv):
    cameraID = 0
    outputVideo = ''
    check = 0
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('main.py -i <cameraID> -o <nameVideo_output>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputvideo> -o null')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            cameraID = arg
        elif opt in ("-o", "--ofile"):
            outputVideo = arg
    if len(cameraID) < 2:
        cameraID = int(cameraID)
        check = 1
    cap = cv2.VideoCapture(cameraID)
    count_total = 0
    count_recognition = 0
    # lưu video
    if check:
        # Set resolutions of frame.
        # convert from float to integer.
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # Create VideoWriter object.
        # and store the output in 'captured_video.avi' file.
        outputVideo = outputVideo + '.avi'
        video_cod = cv2.VideoWriter_fourcc(*'XVID')
        video_output = cv2.VideoWriter(outputVideo,
                                       video_cod,
                                       10,
                                       (frame_width, frame_height))

    while True:
        ret, img = cap.read()
        if check:
            # Write the frame into the file 'captured_video.avi'
            video_output.write(img)
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        name = "Unknown"

        if matches[matchIndex]:
            name = classNames[matchIndex]
            count_recognition += 1
        count_total += 1
        print("true/total: ", count_recognition, '/', count_total)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2-35), (0, 255, 0), 2)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        if ret == True:
            cv2.imshow('Webcam', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break

    # Release handle to the webcam
    cap.release()
    if check:
        video_output.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1:])