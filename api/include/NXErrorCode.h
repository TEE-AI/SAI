/**************************************
FUNC: Engine error code table
DES: 
AUTHOR: WangGaofei
DATA: 2018/11/17
**************************************/
#ifndef _NX_ERROR_CODE_H_
#define _NX_ERROR_CODE_H_

#define NX_RET_SUCCESS                  (0)

#define NX_ERR_UNKNOW                   (-1)

#define NX_ERR_PARAM_NULL               (-101)
#define NX_ERR_PARAM_INVALID            (-102)

#define NX_ERR_NOT_SUPPORT              (-201)

// pixel format not support
#define NX_ERR_IMG_PIX_FMT              (-301)
// image width/height is wrong
#define NX_ERR_IMG_W_H_ERR              (-302)
// image data is NULL
#define NX_ERR_IMG_DATA_NULL            (-303)

// dongle configure file list is invalid
#define NX_ERR_DONGLE_CONF_LIST         (-401)
// dongle get buffer fail from sdk
#define NX_ERR_DONGLE_GET_BUF           (-402)

// open file fail
#define NX_ERR_OPEN_FILE_FAIL           (-501)
// read file fail
#define NX_ERR_READ_FILE_FAIL           (-502)
// write file fail
#define NX_ERR_WRITE_FILE_FAIL          (-503)

// alloc/new memory fail
#define NX_ERR_OUT_OF_MEM               (-601)

#endif // _NX_ERROR_CODE_H_
