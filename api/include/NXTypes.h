/**************************************
FUNC: data types
DES: 
AUTHOR: WangGaofei
DATA: 2018/11/17
**************************************/
#ifndef _NX_TYPES_H_
#define _NX_TYPES_H_
#include <stdlib.h>

typedef int nxi32;
typedef unsigned int nxui32;
typedef char nxi8;
typedef unsigned char nxui8;
typedef short nxi16;
typedef unsigned short nxui16;
typedef long long nxi64;
typedef unsigned long long nxui64;
typedef void nxvoid;
typedef bool nxbool;
typedef float nxreal32;
typedef double nxreal64;
typedef size_t nxsizet;

// engine function return value type, the meaning refer to error code table.
typedef nxi32 NXRet;

#ifndef NULL
#ifdef __cplusplus
#define NULL 0
#else
#define NULL ((void *)0)
#endif
#endif

#endif // _NX_TYPES_H_
