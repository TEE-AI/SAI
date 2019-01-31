/*
*
* Copyright (c) 2018-2019 TEE.COM. All Rights Reserved
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

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

// stick configure file is invalid
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
