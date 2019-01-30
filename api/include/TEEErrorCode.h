/**************************************
FUNC: Engine error code table
DES: 
AUTHOR: WangGaofei
DATA: 2018/11/17
**************************************/
#ifndef _TEE_ERROR_CODE_TABLE_H_
#define _TEE_ERROR_CODE_TABLE_H_

#define TEE_RET_SUCCESS                  (0)

#define TEE_RET_UNKNOW                   (-1)

#define TEE_RET_PARAM_NULL               (-101)
#define TEE_RET_PARAM_INVALID            (-102)

#define TEE_RET_NOT_SUPPORT              (-201)

// pixel format not support
#define TEE_RET_IMG_PIX_FMT              (-301)

// image data is NULL
#define TEE_RET_IMG_DATA_NULL            (-303)

#define TEE_RET_IMAGE_SIZE_ERR                  (-304)
#define TEE_RET_STICK_CREATE_ERR               (-305)

// stick configure file list is invalid
#define TEE_RET_STICK_CONF_LIST         (-401)
// stick get buffer fail from sdk
#define TEE_RET_STICK_GET_BUF           (-402)

// open file fail
#define TEE_RET_OPEN_FILE_FAIL           (-501)
// read file fail
#define TEE_RET_READ_FILE_FAIL           (-502)
// write file fail
#define TEE_RET_WRITE_FILE_FAIL          (-503)

// alloc/new memory fail
#define TEE_RET_OUT_OF_MEM               (-601)

#define TEE_RET_TMP_ERR                  (-602)

#define TEE_RET_DEVICE_INVALID           (-701)

#endif /* _TEE_ERROR_CODE_TABLE_H_ */
