// fake_i3 (11-15): stand-in for the i3system SDK header i3_types.h, only for testing cam_pipe.cpp without a camera (test_cam_pipe_frozen.py).
// Holds the two structs cam_pipe uses, with the members cam_pipe reads. Never install or link this next to the real SDK.
#ifndef I3_TYPES_H
#define I3_TYPES_H

namespace i3 {
  struct SCANINFO { unsigned char bDevCon = 0; unsigned int nCoreID = 0, nProdVer = 0; };   // ScanTE result per device slot
  struct TE_SETTING { unsigned short frameRate, exportable, shutterMode, shutterTime, shutterTemp, highTempMode; };
}

#endif
