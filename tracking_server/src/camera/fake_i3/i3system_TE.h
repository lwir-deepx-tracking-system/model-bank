// fake_i3 (11-15): stand-in for the i3system SDK header i3system_TE.h, only for testing cam_pipe.cpp without a camera (test_cam_pipe_frozen.py).
// It declares what cam_pipe.cpp uses from the SDK (/usr/local/include/i3system on the board) with the same names, values and call forms; fake_i3.cpp
// implements them from a plan in the environment. Never install or link this next to the real SDK: the real cam_pipe is built against /usr/local/include/i3system.
#ifndef I3SYSTEM_TE_H
#define I3SYSTEM_TE_H

#include "i3_types.h"

#define MAX_USB_NUM 32
#define I3_TEMP_SHUTTER   0x0000
#define I3_TIME_SHUTTER   0x0001
#define I3_MANUAL_SHUTTER 0x0002
#define I3_SHUTTER_TIME_MIN 0x0001
#define I3_SHUTTER_TIME_MAX 0x0078
#define I3_SHUTTER_TEMP_MIN 0x0001
#define I3_SHUTTER_TEMP_MAX 0x000A

namespace i3 {
  enum UsbDiagnosticOperation { I3_USB_OPERATION_NONE = 0, I3_USB_OPERATION_SCAN = 1, I3_USB_OPERATION_OPEN = 2, I3_USB_OPERATION_CONTROL = 3,
                                I3_USB_OPERATION_SEND = 4, I3_USB_OPERATION_RECEIVE = 5, I3_USB_OPERATION_INITIALIZE = 6 };
  struct USB_DIAGNOSTIC { int operation; int result; int deviceIndex; };   // result: raw libusb code (-7 timeout, -4 no device) or the SDK's own code (1 = success)

  class TE_A {
  public:
    explicit TE_A(int step);
    void CloseTE();
    int RecvImage(unsigned short* pRecvImage, bool _AGC_On);   // 1 = frame, 4 = no frame (as the SDK)
    int GetImageWidth();
    int GetImageHeight();
    void CalcTemp(unsigned short* _pTempBuf);
    bool ResetMainBoard();
    void SetEmissivity(float _emissivity);
    void GetSetting(TE_SETTING* _pSetting);
    bool SetShutterMode(unsigned short _mode);
    bool SetShutterTime(unsigned short _time);
    bool SetShutterTemperature(unsigned short _temp);
    bool ShutterCalibrationOn();
    float GetFpaTemp();
    float GetShutterPt100Temp();
  private:
    int step_; long sent_ = 0; unsigned short value_ = 0; bool stopped_ = false, closed_ = false;
  };

  extern "C" int ScanTE(SCANINFO* pDevCon);
  extern "C" bool GetLastUsbDiagnostic(USB_DIAGNOSTIC* diagnostic);
  extern "C" const char* GetUsbOperationName(int operation);
  extern "C" const char* GetUsbResultName(int result);
  extern "C" TE_A* OpenTE_A(int hnd_dev);
}

#endif
