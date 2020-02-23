TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        VoteElm.cpp \
        elm.cpp \
        elm_in_elm.cpp \
        functions.cpp \
        main.cpp

INCLUDEPATH += /home/liu/libraries/Eigen

HEADERS += \
  VoteElm.h \
  elm.h \
  elm_in_elm.h \
  functions.h
