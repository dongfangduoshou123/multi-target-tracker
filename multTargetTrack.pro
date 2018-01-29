TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    multtracker.cpp

HEADERS += \
    multtracker.h

INCLUDEPATH += D:\opencv\build\include \
              D:\opencv\build\include\opencv \

CONFIG(debug, debug|release): {
LIBS += -LD:\opencv\build\\x64\vc12\lib \
    -lopencv_core249d \
    -lopencv_imgproc249d \
    -lopencv_highgui249d \
    -lopencv_ml249d \
    -lopencv_video249d \
    -lopencv_features2d249d \
    -lopencv_objdetect249d \
    -lopencv_legacy249d \
    -lopencv_nonfree249d \
    -lopencv_calib3d249d
} else:CONFIG(release, debug|release): {
LIBS += -LD:\opencv\build\x64\vc12\lib \
    -lopencv_core249 \
    -lopencv_imgproc249 \
    -lopencv_highgui249 \
    -lopencv_ml249 \
    -lopencv_video249 \
    -lopencv_features2d249 \
    -lopencv_objdetect249 \
    -lopencv_legacy249 \
    -lopencv_nonfree249 \
    -lopencv_calib3d249
}


win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../libfacedetection-master/lib/ -llibfacedetect-x64
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../libfacedetection-master/lib/ -llibfacedetect-x64
