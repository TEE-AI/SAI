#ifndef TEE_HELMET_ENGINE_H
#define TEE_HELMET_ENGINE_H
#ifdef __cplusplus
extern "C" {
#endif
#ifdef _WIN32
#define NXDLL __declspec(dllimport)
#else
#define NXDLL 
#endif
#ifndef IN
#define IN
#endif

#ifndef OUT
#define OUT
#endif

    /* image pixel format */
    typedef enum {
        eTeeImgFmtBGR = 0, /* BGRBGR... B is the lowest address order */
                           /* new pixel format define from here */
    }TEEImageFmt;

    /* image data */
    typedef struct {
        int w; /* image pixel width */
        int h; /* image pixel height */
        TEEImageFmt fmt; /* image format */
        unsigned char const *data; /* image data */
    }TEEImage;


    typedef enum {
        eTEEDetHead = 0,    /* detect head */
        eTEEDetHelmet = 1, /* detect helmet */
    }eTEEDetType;

    typedef struct {
        int x; // left-top position
        int y;
        int w; // rectangle size
        int h;
        eTEEDetType eType;
    }TEEDetData;

    typedef struct {
        int num; // det data number
        TEEDetData const *vtDetData;
    }TEERet;

    /****************************************************************************
    *   FUN:    Get version.                                                    *
    *   PAR:    none.                                                           *
    *   RET:    major version: (ret&0xffff0000)>>16. minor version: ret&0xffff. *
    ****************************************************************************/
    unsigned int NXDLL TEEGetVersion(void);

    /********************************************************************************************************
    *   FUN:    Create helmet engine.                                                                       *
    *   PAR:    pcConfig -- engine configuration information.                                               *
    *           ppEngine -- *ppEngine is created engine pointer.                                            *
    *   RET:    0 -- success; other -- refer to error code table.                                           *
    ********************************************************************************************************/
    int  NXDLL TEEDetCreateEngine(IN const char* pConfJsonString, OUT void **ppEngine);

    /********************************************************************************************************
    *   FUN:    Detect helmet.                                                                              *
    *   PAR:    pEngine -- helmet engine.                                                                   *
    *           pImg -- image data.                                                                         *
    *           pRet -- detected result.                                                                    *
    *   RET:    0 -- success; other -- refer to error code table.                                           *
    ********************************************************************************************************/
    int NXDLL TEEDetForward(IN void *pEngine, IN TEEImage const *pImg, IN const int nOriginalWidth, IN const int nOriginalHeight, OUT TEERet *pRet);

    /******************************************************************************************
    *   FUN:    Destroy helmet engine.                                                          *
    *   PAR:    ppEngine -- *ppEngine is created engine pointer.                              *
    *   RET:    0 -- success; other -- refer to error code table.                             *
    ******************************************************************************************/
    int  NXDLL TEEDetDestroyEngine(IN OUT void **ppEngine);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // TEE_HELMET_ENGINE_H
